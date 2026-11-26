"""Inspect and stage native ComfyUI H3 video checkpoints without loading tensors."""
import json
import os
from pathlib import Path
import re

from .adaln_assets import asset_path, digest, file_hash
from .file_ops import publish
from .hf_video_catalog import GIB, MARGIN
from .local_model_copy import copy_files, source_record
from .local_video_catalog import _tensor_header
from .long_paths import io_path, plain, resolved
from .monitoring import load_ledger, save


KIND = 'comfy_h3'
PRESETS = {'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}
MAX_SCAN_FILES = 24
MODEL_TARGET = 'assets/model.safetensors'
OWNED = 'prepared/comfy-h3-assets'
DEPENDENCIES = {
    'encoder': ('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 15687142551,
                '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'),
    'video_vae': ('minimax_h3_video_vae_fp16.safetensors', 5207808496,
                  '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'),
    'audio_vae': ('minimax_h3_audio_vae_fp32.safetensors', 605254808,
                  '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
}
FLOATS = {'F16', 'BF16', 'F32'}


def _layout(row):
    header = _tensor_header(row)
    root_shapes = {'video_patch_proj.weight': [5376, 96], 'audio_patch_proj.weight': [5376, 32],
                   'condition_proj.weight': [5376, 5120], 'final_layer.video_out.weight': [96, 5376],
                   'final_layer.audio_out.weight': [32, 5376]}
    if any(header.get(key, ('', []))[1] != shape for key, shape in root_shapes.items()):
        raise ValueError('Expected a native ComfyUI MiniMax H3 video checkpoint.')
    block_ids = {int(match[1]) for key in header if (match := re.match(r'blocks\.(\d+)\.', key))}
    if block_ids != set(range(50)):
        raise ValueError('Expected a 50-block MiniMax H3 video checkpoint.')
    dimensions = header.get('blocks.0.adaln_proj.linear.weight', ('', []))[1]
    if len(dimensions) != 2 or dimensions[1] not in (8, 2688):
        raise ValueError('Unsupported MiniMax H3 AdaLN layout.')
    width = dimensions[1]
    projections = {'attn.qkv_proj': [21504, 5376], 'attn.out_proj': [5376, 7168],
                   'mlp.fc1': [28672, 5376], 'mlp.fc2': [5376, 14336]}
    quantized = False
    for index in range(50):
        block = f'blocks.{index}.'
        for name, shape in (*projections.items(), ('adaln_proj.linear', [96768, width])):
            prefix = block + name
            dtype, dims = header.get(prefix + '.weight', ('', []))
            if dims != shape or dtype not in FLOATS | {'I8'}:
                raise ValueError('Invalid MiniMax H3 projection: ' + prefix)
            if dtype == 'I8':
                marker, marker_shape = header.get(prefix + '.comfy_quant', ('', []))
                if (header.get(prefix + '.weight_scale') != ('F32', [shape[0], 1])
                        or marker != 'U8' or len(marker_shape) != 1 or not 1 <= marker_shape[0] <= 256):
                    raise ValueError('Invalid MiniMax H3 INT8 projection: ' + prefix)
                quantized = True
    return 'comfy_int8' if quantized else 'comfy_float'


def scan(folder):
    directory = Path(plain(resolved(Path(folder))))
    if not io_path(directory).is_dir():
        raise ValueError('Choose a directory with ComfyUI H3 safetensors.')
    rows = []
    # Only direct children of the selected directory are candidates.
    for path in sorted(io_path(directory).glob('*.safetensors'))[:MAX_SCAN_FILES]:
        selection = dict(label=path.stem, presets=[dict(id=key, label=name) for key, name in PRESETS.items()])
        try:
            row = source_record(Path(plain(path)), MODEL_TARGET, root=directory)
            selection['bytes'] = row['bytes']
            if row['bytes'] > 120 * GIB:
                raise ValueError('The local checkpoint is too large.')
            selection.update(format=_layout(row), files=[row], compatible=True)
        except (OSError, ValueError) as error:
            selection.update(compatible=False, compatibility_reason=str(error))
        rows.append(selection)
    return rows


def owned_asset(root, role, checksum):
    if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
        raise ValueError('Invalid ComfyUI H3 content hash.')
    name = 'model.safetensors' if role == 'checkpoint' else DEPENDENCIES[role][0]
    return asset_path(root, f'{OWNED}/{checksum}/{name}')


def owned_stamp(root, path):
    return source_record(path, path.name, root=asset_path(root, OWNED))['fingerprint']


def _asset_receipt(path):
    return path.with_suffix('.verified.json')


def _verified(root, role, checksum):
    path = owned_asset(root, role, checksum)
    try:
        receipt = json.loads(io_path(_asset_receipt(path)).read_bytes())
        return receipt == {'sha256': checksum, 'fingerprint': owned_stamp(root, path)}
    except (OSError, TypeError, ValueError):
        return False


def _publish(root, role, checksum, staged):
    path = owned_asset(root, role, checksum)
    if not _verified(root, role, checksum):
        io_path(path.parent).mkdir(parents=True, exist_ok=True)
        if not io_path(path).exists():
            publish(staged, path)
        else:
            owned_stamp(root, path)
            if file_hash(io_path(path)) != checksum:
                raise ValueError('A stored ComfyUI H3 file failed verification: ' + path.name)
        save(_asset_receipt(path), dict(sha256=checksum, fingerprint=owned_stamp(root, path)), durable=True)
    return path


def _source_index(root, row):
    key = digest([row['source'], row['fingerprint']])
    return asset_path(root, f'{OWNED}/sources/{key}.json')


def reused_checkpoint(root, row):
    try:
        checksum = json.loads(io_path(_source_index(root, row)).read_bytes())['sha256']
        if _verified(root, 'checkpoint', checksum):
            return checksum
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _dependency_source(role, directory, root, machine):
    name, size, checksum = DEPENDENCIES[role]
    if role == 'encoder':
        location = Path(machine['encoder_model_root']) / 'text_encoders'
    else:
        models = next((parent for parent in (directory, *directory.parents)
                       if parent.name.casefold() == 'diffusion_models'), None)
        if models is None:
            raise ValueError('Choose the checkpoint directory under a ComfyUI diffusion_models directory.')
        location = models.parent / 'vae'
    row = source_record(location / name, name, root=location, size=size, sha256=checksum)
    if (role == 'encoder' and Path(row['source']).is_relative_to(root)
            and row['fingerprint']['device'] == io_path(root).stat().st_dev):
        row['link_existing'] = True
    return row


def _link_encoder(root, folder, row):
    """Reuse the verified stock encoder on this installation's own volume."""
    from .provision import verified
    source, staged = Path(row['source']), asset_path(folder, row['target'])
    prior = load_ledger(Path(root) / 'verified-models.json')
    path = io_path(source)
    # The setup ledger stores ordinary paths. I/O uses the extended namespace.
    stamps = {str(path): prior.get(str(source), prior.get(str(path), {}))}
    if not verified(path, row, stamps):
        raise ValueError('The installed H3 encoder failed its SHA-256 check.')
    if source_record(source, row['target'], root=row['root'])['fingerprint'] != row['fingerprint']:
        raise ValueError('The installed H3 encoder changed. Select the checkpoint again.')
    os.link(path, io_path(staged))
    linked = source_record(staged, row['target'], root=folder)['fingerprint']
    if any(linked[key] != row['fingerprint'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns')):
        raise ValueError('The installed H3 encoder changed while it was being linked.')


def _copy_bytes(row):
    return 0 if row.get('link_existing') else row['bytes']


def inspect(selection, preset, root, machine):
    if not selection.get('compatible') or not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError('Choose an inspected ComfyUI H3 model and a workflow preset.')
    model = selection['files'][0]
    actual = source_record(model['source'], model['target'], root=model['root'])
    if actual['fingerprint'] != model['fingerprint']:
        raise ValueError('The checkpoint changed after scanning. Select the directory again.')
    source_dir = Path(model['root'])
    dependencies = {}
    for role, (name, _, checksum) in DEPENDENCIES.items():
        if not _verified(root, role, checksum):
            try:
                dependencies[role] = _dependency_source(role, source_dir, root, machine)
            except (KeyError, OSError, ValueError) as error:
                raise ValueError('The required H3 component could not be read: ' + name) from error
    checksum = reused_checkpoint(root, model)
    model_bytes = 0 if checksum else model['bytes']
    copy_bytes = model_bytes + sum(_copy_bytes(row) for row in dependencies.values())
    return dict(kind=KIND, backend=KIND, source_kind='local', model=preset, preset=preset,
                format=selection['format'], label=selection['label'], compatible=True,
                bytes=model['bytes'], copy_bytes=copy_bytes, required_bytes=copy_bytes + MARGIN,
                files=[model], dependencies=dependencies, reused_checkpoint=checksum)


def stage(root, folder, candidate, progress):
    [(target, model_bytes, checksum)] = candidate['local_hashes']
    model = candidate['files'][0]
    if target != MODEL_TARGET or model_bytes != model['bytes']:
        raise ValueError('Invalid staged ComfyUI H3 checkpoint.')
    path = _publish(root, 'checkpoint', checksum, asset_path(folder, MODEL_TARGET))
    save(_source_index(root, model), dict(sha256=checksum), durable=True)
    assets = {'checkpoint': path.relative_to(root).as_posix()}
    copied = 0 if candidate['reused_checkpoint'] else model_bytes
    for role, (_, _, expected) in DEPENDENCIES.items():
        dependency = candidate['dependencies'].get(role)
        if not _verified(root, role, expected):
            if dependency is None:
                raise ValueError('A shared H3 component changed. Select the checkpoint again.')
            if dependency.get('link_existing'):
                _link_encoder(root, folder, dependency)
            else:
                copy_files(folder, [dependency], lambda done: progress(copied + done))
            _publish(root, role, expected, asset_path(folder, dependency['target']))
        if dependency is not None:
            copied += _copy_bytes(dependency)
        assets[role] = owned_asset(root, role, expected).relative_to(root).as_posix()
    progress(copied)
    return assets


def profile_assets(root, assets):
    if not isinstance(assets, dict) or set(assets) != {'checkpoint', *DEPENDENCIES}:
        raise ValueError('Invalid ComfyUI H3 asset receipt.')
    result = {}
    for role, relative in assets.items():
        path = asset_path(root, relative)
        checksum = path.parent.name if role == 'checkpoint' else DEPENDENCIES[role][2]
        if path != owned_asset(root, role, checksum) or not _verified(root, role, checksum):
            raise ValueError('A stored ComfyUI H3 asset changed. Import the model again.')
        result[role] = str(path)
    return result
