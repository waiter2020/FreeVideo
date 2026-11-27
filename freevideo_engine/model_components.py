"""Versioned replacements for the models FreeVideo actually loads."""
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat

from .adaln_assets import asset_path
from .long_paths import io_path
from .monitoring import save
from .storage import fingerprint


BUILTIN = 'builtin'
ID = re.compile(r'[0-9a-f]{32}')
SLOT_ID = re.compile(r'(?=.{1,80}$)[a-z0-9]+(?:[.-][a-z0-9]+)*')
RECEIPT = 'component.json'


class ComponentError(ValueError):
    pass


@dataclass(frozen=True)
class Slot:
    id: str
    group: str
    label: str
    label_zh: str
    consumers: tuple
    files: dict
    format: str
    validate: object = None
    supported: bool = True
    reason: str = ''
    source: str = ''
    manager: str = ''
    model: str = ''
    component: str = ''
    layout: object = None
    default_status: object = None
    install_default: object = None
    repair_default: object = None
    verify_default: object = None
    default_bytes: int = 0
    license: str = ''


def catalog(root, machine):
    from . import model_components_audio, model_components_generation, model_components_visual
    declarations = [slot for module in (model_components_generation, model_components_visual, model_components_audio)
                    for slot in module.slots(Path(root), machine)]
    result = {slot.id: slot for slot in declarations}
    if len(result) != len(declarations) or any(not SLOT_ID.fullmatch(key) for key in result):
        raise ComponentError('Invalid model component catalog.')
    return result


def current_root():
    from .paths import data_root
    root = data_root()
    if not os.environ.get('FREEVIDEO_HOME') and not io_path(root / 'machine.json').is_file():
        from .comfy_bridge import installation_root
        return installation_root()
    return root


def root_for_model_dir(model_dir):
    if model_dir is not None:
        path = Path(model_dir)
        for parent in (path, *path.parents):
            if io_path(parent / 'machine.json').is_file():
                return parent
    return current_root()


def store(root):
    return _owned_directory(Path(root), 'model-components')


def _owned_directory(root, name):
    folder = asset_path(root, name)
    try:
        entry = io_path(folder).lstat()
    except FileNotFoundError:
        return folder
    if not stat.S_ISDIR(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & 0x400:
        raise ComponentError('The managed model folder is not a regular directory.')
    return folder


def slot_directory(root, slot):
    if not isinstance(slot, str) or not SLOT_ID.fullmatch(slot):
        raise ComponentError('Unknown model component.')
    return _owned_directory(store(root), slot)


def version_directory(root, slot, identity):
    if not isinstance(identity, str) or not ID.fullmatch(identity):
        raise ComponentError('Choose an installed model version.')
    return _owned_directory(slot_directory(root, slot), identity)


def _read(path):
    return json.loads(io_path(path).read_bytes())


def selections(root):
    try:
        value = _read(store(root) / 'selections.json')
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ComponentError('Model component selections could not be read.') from error
    active = value.get('active') if isinstance(value, dict) else None
    if (not isinstance(active, dict) or value.get('schema_version') != 1
            or any(not SLOT_ID.fullmatch(slot) or not isinstance(identity, str) or not ID.fullmatch(identity)
                   for slot, identity in active.items())):
        raise ComponentError('Model component selections are invalid.')
    return active


def choose(root, slot, identity):
    active = selections(root)
    if identity is None:
        active.pop(slot, None)
    else:
        active[slot] = identity
    save(store(root) / 'selections.json', dict(schema_version=1, active=active), durable=True)


def receipt(root, slot, identity):
    folder = version_directory(root, slot, identity)
    try:
        value = _read(asset_path(folder, RECEIPT))
        if (value['schema_version'] != 1 or value['slot'] != slot or value['id'] != identity
                or not isinstance(value['files'], dict) or not value['files']):
            raise ValueError('Invalid receipt')
        for name, info in value['files'].items():
            asset_path(folder, name)
            if (name == RECEIPT or not isinstance(info['stamp'], dict)
                    or not re.fullmatch(r'[0-9a-f]{64}', info['sha256'])):
                raise ValueError('Invalid file record')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ComponentError(f'The installed {slot} version could not be read. Import it again or restore the default.') from error
    return folder, value


def intact(folder, value):
    try:
        return all(fingerprint(asset_path(folder, name)) == row['stamp']
                   for name, row in value['files'].items())
    except (OSError, ValueError):
        return False


def selected_identity(root, slot):
    root = current_root() if root is None else root
    return selections(root).get(slot)


def snapshot(root, slots=None):
    root = current_root() if root is None else root
    selected = selections(root)
    return selected if slots is None else {slot: selected.get(slot, BUILTIN) for slot in slots}


def usage_lock(root, *, shared=False, wait_seconds=0):
    from .locking import runtime_lock
    root = current_root() if root is None else root
    if shared and not store(root).is_dir():
        return nullcontext()
    return runtime_lock(Path(root) / 'model-components-use.lock', inherit=False, shared=shared, wait_seconds=wait_seconds)


def video_snapshot(*, fast=None, two_pass=False, preview=False):
    from .model_components_generation import AUDIO_VAE, PI_SLOTS, SWAP_LORA, VIDEO_VAE
    slots = {AUDIO_VAE, SWAP_LORA, VIDEO_VAE, 'depth-reference'}
    if two_pass:
        slots.add('h3-latent-upscaler')
    if preview:
        slots.update(('video-preview-linear', 'video-preview-decoder'))
    if fast in PI_SLOTS:
        slots.add(PI_SLOTS[fast])
    return {slot: identity for slot, identity in snapshot(None).items() if slot in slots}


def retained_use(output_directory, identity):
    """A working checkpoint can be reopened after a switch away from its model."""
    base = Path(output_directory) / 'FreeVideo'
    for artifact in base.glob('*/*/*.artifacts'):
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(base.resolve()):
            continue
        files = ('first-pass.pt', 'latents.pt', 'refine-input.pt', 'refine-progress.pt')
        if not (any((artifact / name).is_file() for name in files)
                or any((folder / name).is_file() for folder in (artifact / 'attempts').glob('*') for name in files)):
            continue
        report = artifact.with_suffix('.request.json')
        try:
            if report.stat().st_size <= 2 * 1024 * 1024:
                value = _read(report)
                if ((value.get('preview') or value.get('success') is not True)
                        and identity in (value.get('model_components') or {}).values()):
                    return True
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return False


def selected_files(root, slot):
    root = current_root() if root is None else root
    identity = selected_identity(root, slot)
    if identity is None:
        return None, {}
    folder, value = receipt(root, slot, identity)
    if not intact(folder, value):
        raise ComponentError(f'The selected {slot} version is missing or changed. Import it again or restore the default.')
    return folder, {name: asset_path(folder, name) for name in value['files']}


def selected_file(root, slot, name, default):
    folder, files = selected_files(root, slot)
    if folder is None:
        return default
    try:
        return files[name]
    except KeyError as error:
        raise ComponentError(f'The selected {slot} version is missing {name}.') from error


def selected_directory(root, slot, default):
    folder, _ = selected_files(root, slot)
    return default if folder is None else folder


def selected_hash(root, slot, name):
    root = current_root() if root is None else root
    identity = selected_identity(root, slot)
    if identity is None:
        return None
    folder, value = receipt(root, slot, identity)
    if not intact(folder, value) or name not in value['files']:
        raise ComponentError(f'The selected {slot} version is missing or changed.')
    return value['files'][name]['sha256']
