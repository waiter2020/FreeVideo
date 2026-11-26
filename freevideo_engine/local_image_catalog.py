"""Inspect explicitly selected local denoisers without reading their payloads."""
import hashlib
from pathlib import Path

from . import hf_image_catalog as native
from .hf_video_catalog import MARGIN, MAX_METADATA, _json
from .image_gen import image_model, qwen_checkpoint
from .local_model_copy import source_record
from .long_paths import io_path


def _header(record, model):
    try:
        return native._local_header(record['source'], model)
    except OSError as error:
        raise ValueError('Could not read local image weight: ' + Path(record['source']).name) from error


def _inspect_index(selection, destination):
    index = source_record(selection, destination + '/' + native.INDEX, root=selection.parent)
    if index['bytes'] > MAX_METADATA:
        raise ValueError('The MOSS-Image shard index is too large to inspect.')
    try:
        with io_path(index['source']).open('rb') as stream:
            data = stream.read(MAX_METADATA + 1)
    except OSError as error:
        raise ValueError('Could not read the MOSS-Image shard index.') from error
    if len(data) != index['bytes']:
        raise ValueError('The MOSS-Image shard index changed during inspection.')
    try:
        index_json = _json(data)
    except ValueError as error:
        raise ValueError('The MOSS-Image shard index is invalid JSON.') from error
    mapping = native._weight_map(index_json)
    index['sha256'] = hashlib.sha256(data).hexdigest()
    shards = {}
    for name in sorted(set(mapping.values())):
        try:
            shards[name] = source_record(selection.parent / name, destination + '/' + name, root=selection.parent)
        except ValueError as error:
            if isinstance(error.__cause__, FileNotFoundError):
                raise ValueError('Missing MOSS-Image shard: ' + name) from error
            raise
    shapes = native._moss_shapes(mapping, lambda name: _header(shards[name], image_model.MOSS))
    return [index, *shards.values()], shapes


def _incompatible(shapes):
    names = shapes[0]
    if any(name.lower().startswith('lora_') or '.lora_' in name.lower() for name in names):
        return 'This file contains LoRA adapter weights. Choose a complete image denoiser.'
    text_prefixes = ('text_encoder.', 'text_model.', 'language_model.', 'model.language_model.',
                     'model.embed_tokens.', 'model.layers.', 'visual.', 'model.visual.')
    if any(name.startswith(text_prefixes) for name in names):
        return 'This file contains text encoder weights. Choose an image denoiser.'
    return 'The image tensor names, shapes or storage types do not match the selected native model.'


def _compatible(shapes, expected, model):
    if model != image_model.QWEN:
        return shapes == expected
    tensors, config = shapes
    reference, reference_config = expected
    return config == reference_config and qwen_checkpoint.canonical_tensors(tensors) == qwen_checkpoint.canonical_tensors(reference)


def inspect(selection, root, machine, model):
    """Describe only the selected checkpoint and its explicitly indexed sibling shards."""
    selection = Path(selection).absolute()
    name = selection.name.lower()
    indexed = name.endswith('.safetensors.index.json')
    if name.endswith('.gguf'):
        raise ValueError('This selection contains GGUF weights. FreeVideo image switching requires a compatible safetensors denoiser.')
    if indexed and model != image_model.MOSS:
        raise ValueError('Only MOSS-Image accepts a shard index. Choose a single image denoiser safetensors file.')
    if not indexed and not name.endswith('.safetensors'):
        raise ValueError('Choose a native image denoiser safetensors file.')
    pack, path, companions = native._selection(model, machine)
    destination = 'image_gen/' + path
    if indexed:
        files, shapes = _inspect_index(selection, destination)
    else:
        target = destination + '/' + native.SINGLE if model == image_model.MOSS else destination
        file = source_record(selection, target, root=selection.parent)
        files, shapes = [file], _header(file, model)
    expected = native._installed_shapes(pack, path, model)
    if not _compatible(shapes, expected, model):
        raise ValueError(_incompatible(shapes))
    if model == image_model.QWEN:
        native._local_header(files[0]['source'], model, verify_qwen_markers=True)
    total = sum(row['bytes'] for row in files)
    required = total + sum(row['bytes'] for row in companions) + MARGIN
    return dict(label=selection.name, source_kind='local', model=model, kind='image', format=native.FORMATS[model],
                weights=destination, files=files, base_files=companions, bytes=total, copy_bytes=total,
                required_bytes=required, compatible=True)


def validate_import(candidate, pack_dir, machine):
    model = candidate['model']
    if model != image_model.QWEN:
        native.validate_import(candidate, pack_dir, machine)
        return
    stock, path, _ = native._selection(model, machine)
    if candidate['weights'] != 'image_gen/' + path:
        raise ValueError('The selected image checkpoint does not match the native model path.')
    shapes = native._local_header(Path(pack_dir) / path, model, verify_qwen_markers=True)
    if not _compatible(shapes, native._installed_shapes(stock, path, model), model):
        raise ValueError('The staged image checkpoint does not match the verified native image model.')
