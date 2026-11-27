"""Two- and four-step pi-H3 DWD adapters on FreeVideo's installed INT8 weights.

The adapter's pixel head draws the final frames and the audio VAE decodes the sound.
Online LoRA applies the adapter's low-rank factors. The output heads, modulation and
the last four blocks' replacement matrices are changed only in the running engine.

The adapters were trained on plain H3. This integration applies them to FreeVideo's
VDN-derived base.

This module imports no Torch unless it runs as the adapter-preparation child.
"""
import argparse
import json
from pathlib import Path
import sys

FOLDER = 'pi-h3'
CONSTANTS = 'pi_h3_adaln_constants.safetensors'
CONSTANTS_FILE = (19435680, '7949a370323b2a15e13f2e9160cfb1325207cde21b2629643f390be3e32f690b')


def _grid(steps, maximum):
    # The author uses evenly spaced points with a last interval 0.6 times their size.
    interval = maximum / (steps - 1 + .6)
    return tuple(maximum - index * interval for index in range(steps)) + (0.,)


_MODES = {
    'fast2': dict(steps=2, raw_grid=_grid(2, .93), adapter='pi_h3_2step_dwd_adapter_int8_convrot.safetensors',
                  adapter_bytes=2192176240, adapter_sha256='ef98527409eee4eaec8183265f4ee22d265680524237fae7149dbc203e11e739'),
    'fast4': dict(steps=4, raw_grid=_grid(4, .96), adapter='pi_h3_4step_dwd_adapter_int8_convrot.safetensors',
                  adapter_bytes=2192176240, adapter_sha256='a2fa1caae2cd4a12f85cc3eccd2a360fe9dabf430623d47deca619605e96ee6d'),
}
MODES = tuple(_MODES)


def mode_spec(mode):
    try:
        return _MODES[mode]
    except (KeyError, TypeError):
        raise ValueError('Fast mode must be one of ' + ', '.join(MODES)) from None


def unsupported(*, mac=False, gpu_name=None, vendor=None, cuda_missing=False, precision=None):
    """Reason a known setup cannot execute a pi-H3 mode."""
    if mac:
        return 'mac'
    name = str(gpu_name or '').casefold()
    if ((vendor is not None and str(vendor).casefold() != 'nvidia') or cuda_missing
            or any(word in name for word in ('amd', 'radeon', 'instinct'))):
        return 'gpu'
    if precision is not None and not str(precision).startswith('int8'):
        return 'engine'
    return None


def setup_reason(machine=None):
    """Use saved setup facts when available, including on installations without a capability grid."""
    machine = machine or {}
    hardware = machine.get('hardware')
    hardware = hardware if isinstance(hardware, dict) else {}
    capability = hardware.get('capability')
    old_cuda = (isinstance(capability, (tuple, list)) and len(capability) == 2
                and all(type(part) is int for part in capability) and tuple(capability) < (8, 0))
    return unsupported(mac=sys.platform == 'darwin' or machine.get('device_backend') == 'mps',
                       gpu_name=hardware.get('gpu_name') or machine.get('gpu_name'), vendor=machine.get('gpu_vendor'),
                       cuda_missing=old_cuda or ('cuda_version' in hardware and hardware['cuda_version'] in (None, '', 'None')),
                       precision='' if machine.get('video_model') is False else machine.get('prepared_format'))


def validate_setup(reason):
    if reason == 'mac':
        raise ValueError('The fastest mode is not available on Mac')
    if reason == 'gpu':
        raise ValueError('pi-H3 needs an NVIDIA CUDA GPU')
    if reason == 'engine':
        raise ValueError('The fastest mode needs the int8 model')


def files(model_root, mode='fast2', *, engine_root=None):
    """Return the selected, verified runtime files by their basenames."""
    from .fast_pack import files as verified_files
    spec = mode_spec(mode)
    found = verified_files(model_root, mode=mode, engine_root=engine_root)
    return {name: found[f'{FOLDER}/{name}'] for name in (spec['adapter'], CONSTANTS)}


def component_hashes(model_root, mode, *, engine_root=None):
    from .model_components import root_for_model_dir, selected_hash
    from .model_components_generation import PI_SLOTS
    spec = mode_spec(mode)
    root = engine_root or root_for_model_dir(model_root)
    slot = PI_SLOTS[mode]
    return (selected_hash(root, slot, spec['adapter']) or spec['adapter_sha256'],
            selected_hash(root, slot, CONSTANTS) or CONSTANTS_FILE[1])


def request(args, *, original_audio=False):
    """Check a generate command at its entry and set its steps; None without --fast-mode."""
    mode = getattr(args, 'fast_mode', None)
    if mode is None:
        return None
    validate_setup(unsupported(mac=sys.platform == 'darwin'))
    spec = mode_spec(mode)
    upscale_factor = getattr(args, 'upscale_factor', None)
    sampling = bridge_steps(mode,
        preview=getattr(args, 'preview', False), refine_from=getattr(args, 'refine_from', None),
        continue_from=getattr(args, 'continue_from', None) or getattr(args, 'continue_anchor', None),
        upscale_from=getattr(args, 'upscale_from', None), original_audio=original_audio,
        redo_sound=getattr(args, 'redo_sound', None), step_cache=getattr(args, 'step_cache', None),
        upscale_factor=2 if upscale_factor is None else upscale_factor)
    from .paths import installed_model_root
    model_root = installed_model_root()
    paths = files(model_root, mode)
    adapter_hash, constants_hash = component_hashes(model_root, mode)
    args.two_pass, args.base_steps, args.refine_steps = sampling
    return dict(spec, mode=mode, adapter=str(paths[spec['adapter']]), constants=str(paths[CONSTANTS]),
                adapter_sha256=adapter_hash, adapter_bytes=paths[spec['adapter']].stat().st_size,
                constants_sha256=constants_hash)


def bridge_steps(mode, *, preview=False, refine_from=None, continue_from=None, upscale_from=None, step_cache=None,
                 upscale_factor=2, parts=1, original_audio=False, redo_sound=False):
    """(two_pass, base_steps, refine_steps) of a queued request, refused before anything is loaded."""
    spec = mode_spec(mode)
    for value, label in ((preview, 'a preview'), (refine_from, 'finishing a preview'),
                         (continue_from, 'a continued part'), (upscale_from, 'an upscale'),
                         (original_audio, 'the original sound of a video'), (redo_sound, 'redoing the sound'),
                         (step_cache not in (None, '', False, 'none'), 'a step cache'), (upscale_factor != 2, 'another upscale factor'),
                         (parts > 1, 'a video made in parts')):
        if value:
            raise ValueError('The fastest mode cannot be used with ' + label)
    return False, spec['steps'], 3


def adapter_row(spec):
    """The adapter as the first row of the online LoRA request."""
    return dict(path=spec['adapter'], sha256=spec['adapter_sha256'], bytes=spec['adapter_bytes'],
                strength=1., alpha=None, only_factors=True)


def factor_patches(row, targets):
    """The adapter's low-rank pairs. Its whole tensors are applied by fast_tier_runtime.

    The INT8 file names its layers the original way (blocks.N.mlp.fc1, ...); the shared
    LoRA name rules map them and swap fc1's [gate; value] halves, as for any such LoRA.
    """
    from .lora_names import _targets, resolve_target
    from .tensor_io import open_tensors
    suffix = '.lora_A.weight'
    result = []
    with open_tensors(row['path']) as source:
        for key in source.keys():
            if not key.endswith(suffix):
                continue
            other = key[:-len(suffix)] + '.lora_B.weight'
            ((name, split, swap),) = _targets(key[:-len(suffix)])
            target = resolve_target(name, targets)
            a, b = source.get_slice(key).get_shape(), source.get_slice(other).get_shape()
            if target not in targets or targets[target]['shape'] != [b[0], a[1]] or a[0] != b[1]:
                raise ValueError('The fastest mode adapter does not fit this model: ' + target)
            if split is not None:
                raise ValueError('The fastest mode adapter has a fused projection: ' + key)
            result.append(dict(A=key, B=other, target=target, split=None, swap=swap,
                               scale=float(row['strength']), file=row['path']))
    if not result:
        raise ValueError('The fastest mode adapter has no low-rank factors')
    return result


def main():
    """The LoRA preparation child for a fast request.

    The folder identity hashes lora_cache.py and lora_online_cache.py, so a change
    there would rebuild every installed LoRA folder. Only this process indexes the
    adapter row differently; user LoRA rows take the existing route.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    value = json.loads(args.request.read_text(encoding='utf-8'))
    if value.get('mode') != 'online':
        raise ValueError('The fastest mode needs the online LoRA placement')
    import torch
    from . import lora_online_cache as online
    from .monitoring import save
    torch.set_num_threads(8)
    existing = online.index_adapter

    def index_adapter(row, targets, skipped=None):
        return factor_patches(row, targets) if row.get('only_factors') else existing(row, targets, skipped)
    online.index_adapter = index_adapter
    output, report = online.prepare(value['cache'], value['adapters'])
    save(value['result'], {'cache': str(output), 'report': report})


if __name__ == '__main__':
    main()
