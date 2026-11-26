"""Inspect native image encoder and VAE checkpoints against the installed weights."""
import hashlib
import json
from pathlib import Path

from . import hf_auth, hf_image_catalog as native, hf_video_catalog as hub
from .adaln_assets import asset_path
from .image_gen import image_model, image_pack
from .local_model_copy import source_record
from .long_paths import io_path


TEXT_ENCODER = 'text_encoder'
VAE = 'vae'
FORMATS = {
    image_model.QWEN: {TEXT_ENCODER: 'int8_convrot', VAE: 'bf16'},
    image_model.H3: {TEXT_ENCODER: 'nvfp4_awq', VAE: 'fp16'},
    image_model.MOSS: {TEXT_ENCODER: 'bf16', VAE: 'fp32'},
}
MOSS_FILES = {TEXT_ENCODER: 'model.safetensors', VAE: 'diffusion_pytorch_model.safetensors'}


def stock_paths(machine, model, component):
    """The built-in checkpoint file, or the Diffusers/Transformers component directory."""
    if model not in FORMATS or component not in FORMATS[model]:
        raise ValueError('Choose a supported native image model component.')
    if (model == image_model.H3 and component == TEXT_ENCODER
            and machine.get('encoder_model_root') and machine.get('encoder')):
        return Path(machine['encoder_model_root']) / 'text_encoders' / machine['encoder']
    model_root = machine.get('model_root')
    if not model_root:
        raise ValueError('Install the built-in image model component first.')
    rows = image_pack.rows(group=native.GROUPS[model])
    row = next(row for row in rows if row['component'] == component and row['file'].endswith('.safetensors'))
    path = image_pack.target(row, model_root)
    return path.parent if model == image_model.MOSS else path


def _checkpoint(path, *, verify_markers=False):
    markers = {} if verify_markers else None
    try:
        shapes, config = native._local_header(path, None, markers=markers, parse_config=True)
    except OSError as error:
        raise ValueError('Could not read image model component: ' + Path(path).name) from error
    return shapes, config, markers


def _local_index(path):
    try:
        with io_path(path).open('rb') as stream:
            data = stream.read(hub.MAX_METADATA + 1)
    except OSError as error:
        raise ValueError('Could not read the MOSS-Image component shard index.') from error
    if len(data) > hub.MAX_METADATA:
        raise ValueError('The MOSS-Image component shard index is too large to inspect.')
    return native._weight_map(hub._json(data)), data


def _combined(mapping, read):
    """Recombine shards independent of the upstream packing order."""
    by_file = {}
    for name, file in mapping.items():
        by_file.setdefault(file, set()).add(name)
    shapes = {}
    for file, expected in sorted(by_file.items()):
        found, config, _ = read(file)
        if set(found) != expected or config:
            raise ValueError('MOSS-Image component shards disagree with the index or built-in configuration.')
        shapes.update(found)
    return shapes, {}, None


def _signature(path, model, component, *, verify_markers=False):
    if model != image_model.MOSS:
        return _checkpoint(path, verify_markers=verify_markers and component == TEXT_ENCODER)
    single = MOSS_FILES[component]
    index = path / (single + '.index.json')
    if component == TEXT_ENCODER and io_path(index).is_file():
        mapping, _ = _local_index(index)
        return _combined(mapping, lambda file: _checkpoint(asset_path(path, file)))
    return _checkpoint(path / single)


def _qwen_language_keys(tensors):
    normalized = {}
    for key, signature in tensors.items():
        if key.startswith('lm_head.'):
            continue
        if key.startswith('model.language_model.'):
            key = 'model.' + key.removeprefix('model.language_model.')
        if key in normalized:
            raise ValueError('The Qwen3-VL checkpoint contains duplicate language tensor names.')
        normalized[key] = signature
    return normalized


def _h3_plain_shapes(shapes, stock_shapes):
    # Non-AWQ NVFP4 omits the paired activation smoothing tensors throughout the encoder.
    smoothing = '.pre_quant_scale'
    if (any(name.endswith(smoothing) for name in shapes)
            or not any(name.endswith(smoothing) for name in stock_shapes)):
        return None
    expected = {name: signature for name, signature in stock_shapes.items() if not name.endswith(smoothing)}
    for name, signature in shapes.items():
        if (name.endswith('.weight_scale_2') and expected.get(name) == ('F32', ())
                and signature == ('F32', (1,))):
            expected[name] = signature
        elif (name == 'model.embed_tokens.weight_scale' and name in expected
              and signature == ('F32', ())):
            expected[name] = signature
    return expected


def _h3_plain_markers_match(markers, stock_markers):
    if markers.keys() != stock_markers.keys():
        return False
    for name, stock in stock_markers.items():
        marker = markers[name]
        if stock.get('format') == 'nvfp4':
            if (not isinstance(marker, dict) or marker.get('format') != 'nvfp4'
                    or marker.keys() - {'format', 'full_precision_matrix_mult'}
                    or type(marker.get('full_precision_matrix_mult', False)) is not bool):
                return False
        elif marker != stock:
            return False
    return True


def _check(found, expected, model, component):
    shapes, config, markers = found
    stock_shapes, stock_config, stock_markers = expected
    format = FORMATS[model][component]
    qwen_encoder = model == image_model.QWEN and component == TEXT_ENCODER
    if qwen_encoder:
        shapes, stock_shapes = _qwen_language_keys(shapes), _qwen_language_keys(stock_shapes)
    elif model == image_model.H3 and component == TEXT_ENCODER:
        plain = _h3_plain_shapes(shapes, stock_shapes)
        if plain is not None:
            stock_shapes, format = plain, 'nvfp4'
    if shapes != stock_shapes or config != stock_config:
        raise ValueError('The tensor names, shapes, storage types or configuration do not match the built-in ' +
                         ('text encoder.' if component == TEXT_ENCODER else 'VAE.'))
    if markers is not None:
        if qwen_encoder:
            markers, stock_markers = _qwen_language_keys(markers), _qwen_language_keys(stock_markers)
        matches = (_h3_plain_markers_match(markers, stock_markers) if format == 'nvfp4'
                   else json.dumps(markers, sort_keys=True) == json.dumps(stock_markers, sort_keys=True))
        if not matches:
            raise ValueError('The text encoder quantization markers do not match the built-in checkpoint.')
    return format


def _selection(machine, model, component, *, verify_markers):
    path = stock_paths(machine, model, component)
    target = '.' if model == image_model.MOSS else path.name
    return target, _signature(path, model, component, verify_markers=verify_markers)


def _candidate(model, component, target, files, format):
    total = sum(row['bytes'] for row in files)
    return dict(model=model, component=component, kind='image', format=format, weights=target,
                files=files, base_files=[], bytes=total, required_bytes=total + hub.MARGIN, compatible=True)


def _target(source, model, component, single):
    name = source.lower()
    if name.endswith('.gguf'):
        raise ValueError('Choose compatible safetensors weights for this component. GGUF is unsupported.')
    if name.endswith('.safetensors.index.json'):
        if model != image_model.MOSS or component != TEXT_ENCODER:
            raise ValueError('This component requires a single safetensors checkpoint.')
        return MOSS_FILES[component] + '.index.json'
    if not name.endswith('.safetensors'):
        raise ValueError('Choose a native image component safetensors file or supported shard index.')
    return MOSS_FILES[component] if model == image_model.MOSS else single


def inspect_local(selection, root, machine, model, component):
    """Inspect the chosen local weights and only the shards named by its index."""
    selection = Path(selection).absolute()
    target, expected = _selection(machine, model, component, verify_markers=True)
    file = source_record(selection, _target(selection.name, model, component, target), root=selection.parent)
    if file['target'].endswith('.index.json'):
        mapping, data = _local_index(file['source'])
        if len(data) != file['bytes']:
            raise ValueError('The MOSS-Image component shard index changed during inspection.')
        file['sha256'] = hashlib.sha256(data).hexdigest()
        shards = {name: source_record(selection.parent / name, name, root=selection.parent)
                  for name in sorted(set(mapping.values()))}
        found = _combined(mapping, lambda name: _checkpoint(shards[name]['source']))
        files = [file, *shards.values()]
    else:
        found = _checkpoint(file['source'], verify_markers=component == TEXT_ENCODER and model != image_model.MOSS)
        files = [file]
    format = _check(found, expected, model, component)
    row = _candidate(model, component, target, files, format)
    return dict(row, label=selection.name, source_kind='local', copy_bytes=row['bytes'])


def _choices(entries, prefix, model, component, target):
    prefix = prefix or ''
    if prefix.endswith(('.safetensors', '.safetensors.index.json', '.gguf')):
        return [prefix]
    prefix = prefix.rstrip('/') + '/' if prefix else ''
    canonical = MOSS_FILES[component] if model == image_model.MOSS else target
    roles = {'text_encoder', 'text_encoders'} if component == TEXT_ENCODER else {'vae'}
    extensions = ('.safetensors', '.safetensors.index.json') if model == image_model.MOSS and component == TEXT_ENCODER else ('.safetensors',)
    choices = [name for name in entries if name.startswith(prefix) and name.endswith(extensions)]
    return sorted(choices, key=lambda name: (name.rsplit('/', 1)[-1] not in (canonical, canonical + '.index.json'),
                                             not roles.intersection(name.split('/')[:-1]),
                                             not name.endswith('.index.json'), len(name), name))[:native.MAX_CHOICES]


def _remote_component(repo, revision, entries, name, model, component, target, token):
    file = hub._row(repo, revision, entries, name, _target(name, model, component, target))

    def header(row):
        shapes, config = native._remote_header(row, None, token, parse_config=True)
        return shapes, config, None

    if not file['target'].endswith('.index.json'):
        return [file], header(file)
    index, checksum = hub._read(repo, revision, entries, name, token)
    file.pop('git_blob', None)
    file['sha256'] = checksum
    mapping = native._weight_map(index)
    folder = name.rpartition('/')[0]
    shards = {part: hub._row(repo, revision, entries, hub._at(folder, part), part) for part in sorted(set(mapping.values()))}
    return [file, *shards.values()], _combined(mapping, lambda part: header(shards[part]))


def inspect(reference, root, machine, model, component, *, token=None):
    """Inspect a Hugging Face repository or file pinned to the resolved commit."""
    repo, revision, prefix = native._reference(reference, None, None)
    token = hf_auth.validate(token or '')
    target, expected = _selection(machine, model, component, verify_markers=False)
    info, entries = hub._info(repo, revision, token)
    repo, revision = info['id'], info['sha']
    for name in _choices(entries, prefix, model, component, target):
        try:
            files, found = _remote_component(repo, revision, entries, name, model, component, target, token)
            format = _check(found, expected, model, component)
        except ValueError:
            if name == prefix:
                raise
        else:
            break
    else:
        raise ValueError('No compatible image model component was found. Paste the exact checkpoint or shard index URL.')
    license_file = 'LICENSE' if 'LICENSE' in entries else None
    if license_file:
        files.append(hub._row(repo, revision, entries, license_file, 'LICENSE'))
    label = name.rsplit('/', 1)[-1]
    row = _candidate(model, component, target, files, format)
    return dict(row, id=native._identifier(repo, revision, name, model + '/' + component),
                repo=repo, revision=revision, path=name, subfolder=prefix, label=label,
                download_bytes=row['bytes'], license=hub._license(info), gated=bool(info.get('gated')),
                license_url=hub.HUB + '/' + repo + '/blob/' + revision + '/LICENSE' if license_file
                else hub.HUB + '/' + repo + '/tree/' + revision)


def validate_import(candidate, folder, machine):
    """Recheck the staged bytes and marker payloads before the runtime smoke test."""
    model, component = candidate['model'], candidate['component']
    target, expected = _selection(machine, model, component, verify_markers=True)
    if candidate.get('weights') != target:
        raise ValueError('The staged checkpoint does not match the native image component path.')
    path = Path(folder) if target == '.' else asset_path(folder, target)
    found = _signature(path, model, component, verify_markers=True)
    if _check(found, expected, model, component) != candidate.get('format'):
        raise ValueError('The staged checkpoint format changed after inspection.')
