"""Test an assembled image pack with the same runner used for ordinary requests."""
import argparse
import json
import os
from pathlib import Path

from . import image_model

PROMPT = 'A blue ceramic cup on a wooden table in soft daylight.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--machine', type=Path, required=True)
    parser.add_argument('--model', choices=image_model.MODELS, required=True)
    parser.add_argument('--components-json', type=Path, help='Private resolved component selection')
    parser.add_argument('--scenario', choices=('text', 'reference', 'mask'), default='text')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.scenario == 'mask' and args.model != image_model.QWEN:
        parser.error('Only Qwen-Image supports the mask scenario.')
    if args.scenario == 'reference' and args.model == image_model.MOSS:
        parser.error('MOSS-Image supports only the text scenario.')
    if args.components_json is None and args.pack is None:
        parser.error('Provide --pack or --components-json.')
    from PIL import Image, ImageDraw
    from .h3_runner import STUB_ENV as H3_STUB
    from .moss_runner import STUB_ENV as MOSS_STUB
    from .worker import load_h3_runner, load_moss_runner, load_runner
    # Protocol tests can fake runner output. An import test has to exercise the candidate.
    os.environ.pop(H3_STUB, None)
    os.environ.pop(MOSS_STUB, None)
    machine = json.loads(args.machine.read_text(encoding='utf-8'))
    factories = {image_model.QWEN: load_runner, image_model.H3: load_h3_runner, image_model.MOSS: load_moss_runner}
    pack, options = args.pack, {}
    if args.components_json is not None:
        resolved = json.loads(args.components_json.read_text(encoding='utf-8'))
        if resolved['model'] != args.model:
            parser.error('The resolved components do not match --model.')
        pack, options = Path(resolved['pack']), {'components': resolved['components']}
    dimensions = image_model.size(args.model, '1:1') or (256, 256)
    inputs = {}
    if args.scenario != 'text':
        width, height = dimensions
        reference = Image.new('RGB', dimensions, '#e6cfab')
        ImageDraw.Draw(reference).ellipse((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill='#2266aa')
        inputs['reference_images'] = [reference]
        if args.scenario == 'mask':
            mask = Image.new('L', dimensions)
            ImageDraw.Draw(mask).rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=255)
            inputs['edit_mask'] = mask
    runner = factories[args.model](pack, machine.get('device_backend', 'cuda'), **options)
    try:
        if args.model in (image_model.H3, image_model.MOSS):
            inputs['setup'] = dict(machine, smoke=True)
        result = runner.generate(PROMPT, width=dimensions[0], height=dimensions[1], seed=0, **inputs)
        with args.out.open('xb') as output:
            result.save(output, format='PNG')
    finally:
        runner.close()
    with Image.open(args.out) as checked:
        checked.load()
        if checked.format != 'PNG' or checked.size != dimensions:
            raise ValueError('The candidate model did not generate the expected test image.')


if __name__ == '__main__':
    main()
