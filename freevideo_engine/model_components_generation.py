"""Model component inventory and runtime paths for image and video generation."""
from functools import lru_cache
import json
from pathlib import Path


VIDEO_VAE = 'h3-video-vae'
AUDIO_VAE = 'h3-audio-vae'
NATIVE_VIDEO_VAE = 'h3-comfy-video-vae'
NATIVE_AUDIO_VAE = 'h3-comfy-audio-vae'
SWAP_LORA = 'h3-character-swap'
PI_SLOTS = {'fast2': 'pi-h3-2-step', 'fast4': 'pi-h3-4-step'}


def vae_directory(base, *, audio=False):
    from .model_components import root_for_model_dir, selected_directory
    directory = Path(base) / ('audio_vae' if audio else 'vae')
    return selected_directory(root_for_model_dir(base), AUDIO_VAE if audio else VIDEO_VAE, directory)


def _tensor_layout(path):
    from .local_video_catalog import _tensor_header
    from .long_paths import io_path
    path = io_path(path)
    return _tensor_header({'source': str(path), 'target': path.name, 'bytes': path.stat().st_size})


def _weight_layout(directory):
    weights = {}
    for path in sorted(directory.glob('*.safetensors')):
        layout = _tensor_layout(path)
        if weights.keys() & layout.keys():
            raise ValueError('Duplicate VAE tensors across shards.')
        weights.update(layout)
    if not weights:
        raise ValueError('No VAE tensors were found.')
    return weights


def _vae_layout(names):
    paths = {name for name in names if '/' not in name and (name.endswith('.safetensors')
             or name in ('config.json', 'diffusion_pytorch_model.safetensors.index.json'))}
    if 'config.json' not in paths or not any(name.endswith('.safetensors') for name in paths):
        raise ValueError('Choose a Diffusers VAE directory with config.json and safetensors weights.')
    return paths


def _vae_config(path):
    from .local_video_catalog import _metadata
    value, _ = _metadata(path, 'VAE configuration')
    if not isinstance(value, dict):
        raise ValueError('Invalid VAE configuration.')
    return {key: setting for key, setting in value.items() if not key.startswith('_') or key == '_class_name'}


def _validate_vae(builtin, candidate):
    if _vae_config(builtin / 'config.json') != _vae_config(candidate / 'config.json'):
        raise ValueError('The VAE configuration does not match the H3 architecture used by this installation.')
    if _weight_layout(builtin) != _weight_layout(candidate):
        raise ValueError('The VAE tensor names, dimensions or types do not match the installed H3 VAE.')


def _validate_single(staged, name, reference):
    if _tensor_layout(staged / name) != _tensor_layout(reference):
        raise ValueError('The replacement tensor layout does not match ' + reference.name + '.')


def _validate_fast(staged, defaults, rows):
    from .adaln_assets import file_hash
    for name, reference in defaults.items():
        if reference.is_file():
            _validate_single(staged, name, reference)
        elif file_hash(staged / name) != rows[name]['sha256']:
            raise ValueError('Install the pinned pi-H3 pack first to validate a different adapter layout.')


@lru_cache(maxsize=1)
def _default_vae_records():
    records = json.loads(Path(__file__).with_name('model_files.json').read_text(encoding='utf-8'))
    return {folder: {row['file'].rsplit('/', 1)[-1]: row for row in records
                     if row['file'].startswith(f'h3-base/{folder}/')} for folder in ('vae', 'audio_vae')}


def _verify_default(files, records):
    from .adaln_assets import file_hash
    from .long_paths import io_path
    for name, path in files.items():
        row = records[name]
        path = io_path(path)
        if (not path.is_file() or path.stat().st_size != row['bytes']
                or file_hash(path) != (row.get('download_sha256') or row.get('sha256'))):
            raise ValueError('The built-in file failed verification: ' + name)


def native_vae_files(root, assets):
    from .local_comfy_h3_catalog import DEPENDENCIES
    from .model_components import selected_file
    files = dict(assets)
    for role, slot in (('video_vae', NATIVE_VIDEO_VAE), ('audio_vae', NATIVE_AUDIO_VAE)):
        if role in files:
            files[role] = str(selected_file(root, slot, DEPENDENCIES[role][0], files[role]))
    return files


def _native_defaults(root, machine):
    from .hf_video_models import ACTIVE_KEY, _receipt, _verified_comfy_profile, profile_folder
    active = machine.get(ACTIVE_KEY) or {}
    profiles = root / 'prepared' / 'huggingface'
    folders = sorted(profiles.glob('*'))
    if active.get('kind') == 'comfy_h3' or active.get('backend') == 'comfy_h3':
        folders.insert(0, profiles / active['id'])
    for folder in folders:
        try:
            record, cache = _receipt(profile_folder(root, folder.name))
            if cache is None:
                return _verified_comfy_profile(root, record, machine)['assets']
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {}


def slots(root, machine):
    from . import fast_pack, fast_tier, image_components, swap_lora
    from .hf_image_models import FAMILIES
    from .image_gen import image_model
    from .local_comfy_h3_catalog import DEPENDENCIES
    from .model_components import Slot

    model_root = Path(machine.get('model_root') or root / 'models')
    base = Path(machine.get('base') or model_root / 'h3-base')
    video = machine.get('video_model', True) and bool(machine.get('base'))
    native = _native_defaults(root, machine)
    result = [
        Slot('h3-video-model', 'video', 'MiniMax H3 video', 'MiniMax H3 视频', ('Video generation',),
             {}, 'VDN H3 raw / prepared cache', manager='video', supported=bool(video)),
        Slot('h3-comfy-profile', 'video', 'Native ComfyUI H3 profile', 'ComfyUI 原生 H3 预设',
             ('FL2VA', 'Ref2VA'), {}, 'ComfyUI H3 Safetensors', manager='video', component='comfy_h3',
             supported=bool(video)),
    ]
    for slot, name, label, zh in ((VIDEO_VAE, 'vae', 'H3 video VAE', 'H3 视频 VAE'),
                                  (AUDIO_VAE, 'audio_vae', 'H3 audio VAE', 'H3 音频 VAE')):
        builtin = base / name
        records = _default_vae_records()[name]
        files = {filename: builtin / filename for filename in records}
        ready = bool(video and all(path.is_file() for path in files.values()))
        result.append(Slot(slot, 'video', label, zh, ('Video generation', 'Reference conditioning'),
                           files, 'Diffusers Safetensors',
                           validate=lambda staged, directory=builtin: _validate_vae(directory, staged),
                           supported=ready, reason='' if ready else 'Install the built-in H3 video model first.',
                           source='OpenVDN/vdn-minimax-h3', layout=_vae_layout,
                           verify_default=lambda paths=files, expected=records: _verify_default(paths, expected),
                           default_bytes=sum(row['bytes'] for row in records.values())))

    for slot, role, label, zh in ((NATIVE_VIDEO_VAE, 'video_vae', 'Native H3 video VAE', '原生 H3 视频 VAE'),
                                   (NATIVE_AUDIO_VAE, 'audio_vae', 'Native H3 audio VAE', '原生 H3 音频 VAE')):
        filename, size, checksum = DEPENDENCIES[role]
        builtin = Path(native.get(role) or model_root / 'vae' / filename)
        ready = builtin.is_file() and role in native
        files = {filename: builtin}
        records = {filename: dict(bytes=size, sha256=checksum)}
        result.append(Slot(slot, 'video', label, zh, ('Native ComfyUI H3',), {filename: builtin}, 'Safetensors',
                           validate=lambda staged, name=filename, path=builtin: _validate_single(staged, name, path),
                           supported=ready, reason='' if ready else 'Import a native ComfyUI H3 profile first.',
                           default_bytes=size, source='Comfy-Org/MiniMax-H3',
                           verify_default=lambda paths=files, expected=records: _verify_default(paths, expected)))

    pack = fast_pack.catalog()
    fast_root = fast_pack._model_folder(dict(machine, model_root=str(model_root)))
    for mode, slot in PI_SLOTS.items():
        spec = fast_tier.mode_spec(mode)
        rows = {row['file'].rsplit('/', 1)[-1]: row for row in fast_pack.rows(pack, mode=mode) if row.get('engine')}
        files = {name: fast_pack._target(fast_root, row) for name, row in rows.items()}
        reason = fast_tier.setup_reason(machine)
        result.append(Slot(slot, 'video', f"pi-H3 {spec['steps']}-step adapter", f"pi-H3 {spec['steps']} 步适配器",
                           ('pi-H3 generation',), files, 'ConvRot INT8 Safetensors',
                           validate=lambda staged, defaults=files, expected=rows: _validate_fast(staged, defaults, expected),
                           supported=reason is None and bool(video), reason=reason or '', source=pack['repo'], license=pack['license'],
                           default_status=lambda mode=mode: dict(installed=fast_pack.default_present(fast_root, pack, mode=mode)),
                           install_default=lambda progress, check, mode=mode: fast_pack.install(
                               fast_root, lambda event: progress(event['done'], event['total']), check,
                               engine_root=root, spec=pack, mode=mode),
                           verify_default=lambda mode=mode: fast_pack.default_files(fast_root, pack, mode=mode),
                           default_bytes=sum(row['bytes'] for row in rows.values())))

    swap = swap_lora.catalog()
    try:
        folder = swap_lora.lora_folder()
    except (ImportError, IndexError, KeyError):
        folder = Path(machine.get('comfy_root') or root) / 'models' / 'loras'
    filename = swap['name']
    records = {row['target']: row for row in swap['files']}
    files = {name: folder / name for name in records}
    result.append(Slot(SWAP_LORA, 'video', 'Replace person LoRA', '替换人物 LoRA', ('Replace person',),
                       files, 'H3 LoRA Safetensors',
                       validate=lambda staged: swap_lora.validate_component(staged / filename),
                       install_default=lambda progress, check: swap_lora.install(
                           folder, (), lambda event: progress(event['done'], event['total']), check,
                           engine_root=root, verify_existing=True),
                       verify_default=lambda paths=files, expected=records: _verify_default(paths, expected),
                       default_bytes=sum(row['bytes'] for row in swap['files']), source=swap['repo'], license=swap['license']))

    for model, (label, _) in FAMILIES.items():
        for role, role_zh in (('dit', 'DiT 去噪器'), ('text_encoder', '文本编码器'), ('vae', 'VAE')):
            shared = model == image_model.H3 and role == 'text_encoder'
            title = 'Shared H3 text encoder' if shared else f'{label} {image_components.LABELS[role]}'
            title_zh = 'H3 共享文本编码器' if shared else f'{label} {role_zh}'
            files = {}
            try:
                path = Path(image_components._stock(dict(machine, model_root=str(model_root)), model, role)['path'])
                files = {entry.name: entry for entry in path.iterdir() if entry.is_file()} if path.is_dir() else {path.name: path}
            except (OSError, ValueError, KeyError, RuntimeError, StopIteration):
                pass
            consumers = ('Image generation', 'VDN H3 video', 'Native ComfyUI H3') if shared else ('Image generation',)
            result.append(Slot(f'{model}-{role.replace("_", "-")}', 'image', title, title_zh, consumers, files,
                               'Safetensors / supported weight formats', manager='image', model=model, component=role))
    return result
