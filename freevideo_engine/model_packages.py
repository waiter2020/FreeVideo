"""Named bundles of replaceable image and pi-H3 weights.

A package stores references, not another copy of the weights. Image roles point at an
image component id (24 hex) or ``builtin``. pi-H3 roles point at one model-component
version (32 hex) or ``builtin``. Steps and sigma grids stay on the package type.
"""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import secrets
import sys

from .long_paths import io_path
from .monitoring import save


FILE = 'model-packages.json'
LOCK_NAME = 'model-packages.lock'
BUSY_IMAGE = 'Another task is using the image model package; try again when it finishes.'
ACTIVE_IMAGE = 'This image package is in use. Activate another package first.'
IMAGE_ROLES = ('dit', 'text_encoder', 'vae')
FAST_ROLES = ('adapter', 'constants')
ROLE_TEXT = {
    'dit': ('Denoiser', '去噪器'),
    'text_encoder': ('Text encoder', '文本编码器'),
    'vae': ('VAE', 'VAE'),
    'adapter': ('Adapter', '适配器'),
    'constants': ('Constants', '常量'),
}
# The same sentences the picture-model importer shows, keyed by family then role.
FORMATS = {
    'qwen-image-2.1-turbo': {
        'dit': ('Qwen-Image 2.1 Turbo INT8 ConvRot, one .safetensors file.',
                'Qwen-Image 2.1 Turbo INT8 ConvRot，单个 .safetensors 文件。'),
        'text_encoder': ('Qwen3-VL 8B, INT8 ConvRot text weights (group size 256) and a complete BF16 vision tower in one .safetensors file.',
                         'Qwen3-VL 8B，INT8 ConvRot 文本权重（组大小 256）和完整 BF16 视觉塔须在同一个 .safetensors 文件中。'),
        'vae': ('Qwen-Image 2.1 BF16 VAE, one .safetensors file.', 'Qwen-Image 2.1 BF16 VAE，单个 .safetensors 文件。'),
    },
    'minimax-h3-image': {
        'dit': ('MiniMax H3 Image INT8 ConvRot, one .safetensors file.',
                'MiniMax H3 Image INT8 ConvRot，单个 .safetensors 文件。'),
        'text_encoder': ('MiniMax H3 Qwen3-VL 32B in ComfyUI NVFP4 AWQ or plain NVFP4 format, one .safetensors file.',
                         '支持 ComfyUI 格式的 MiniMax H3 Qwen3-VL 32B NVFP4 AWQ 或普通 NVFP4，单个 .safetensors 文件。'),
        'vae': ('MiniMax H3 Image FP16 VAE, one .safetensors file.', 'MiniMax H3 Image FP16 VAE，单个 .safetensors 文件。'),
    },
    'moss-image-sft': {
        'dit': ('MOSS-Image-SFT BF16 denoiser, a .safetensors file or a Diffusers shard index.',
                'MOSS-Image-SFT BF16 去噪器，支持 .safetensors 文件或 Diffusers 分片索引。'),
        'text_encoder': ('MOSS-Image-SFT BF16 text encoder, a .safetensors file or a Transformers shard index.',
                         'MOSS-Image-SFT BF16 文本编码器，支持 .safetensors 文件或 Transformers 分片索引。'),
        'vae': ('MOSS-Image-SFT FP32 VAE, one .safetensors file.', 'MOSS-Image-SFT FP32 VAE，单个 .safetensors 文件。'),
    },
}
VERSION = 2
LEGACY_KEYS = {'schema_version', 'active', 'legacy_fast', 'shared_encoder', 'packages'}
TOP_KEYS = LEGACY_KEYS | {'image_defaults'}
PACKAGE_KEYS = {'id', 'type', 'name', 'builtin', 'bindings', 'parameters'}
PACKAGE_ID = re.compile(r'pkg_[0-9a-f]{32}')
_TYPES = None


class PackageError(ValueError):
    """A model package could not be read or changed. The previous package stays in use."""


def builtin_package_id(type_id):
    return 'builtin:' + type_id


def valid_package_id(value):
    return isinstance(value, str) and (PACKAGE_ID.fullmatch(value) is not None
                                       or value.startswith('builtin:') and value[8:] in type_map())


def type_map():
    """Package types shipped with this version. Users choose one; they do not define one."""
    global _TYPES
    if _TYPES is not None:
        return _TYPES
    from .image_gen import image_model
    from .fast_tier import mode_spec

    def image(family, label, label_zh):
        return dict(category='image', family=family, label=label, label_zh=label_zh,
                    roles=IMAGE_ROLES, parameters=())

    def video(mode, label, label_zh):
        spec = mode_spec(mode)
        return dict(category='video', mode=mode, label=label, label_zh=label_zh, roles=FAST_ROLES, parameters=(
            dict(key='steps', kind='int', label='Steps', label_zh='步数', editable=False, default=spec['steps']),
            dict(key='grid', kind='floats', label='Sigma grid', label_zh='sigma 网格', editable=False,
                 default=tuple(spec['raw_grid'])),
        ))

    _TYPES = {
        'qwen-image-2.1': image(image_model.QWEN, 'Qwen-Image 2.1', 'Qwen-Image 2.1'),
        'minimax-h3-image': image(image_model.H3, 'MiniMax H3 Image', 'MiniMax H3 Image'),
        'moss-image-sft': image(image_model.MOSS, 'MOSS-Image', 'MOSS-Image'),
        'pi-h3-2-step': video('fast2', 'pi-H3 2-step', 'pi-H3 2 步'),
        'pi-h3-4-step': video('fast4', 'pi-H3 4-step', 'pi-H3 4 步'),
    }
    return _TYPES


def type_for_family(family):
    for type_id, spec in type_map().items():
        if spec['category'] == 'image' and spec['family'] == family:
            return type_id
    raise PackageError('Choose a supported image model.')


def type_for_mode(mode):
    for type_id, spec in type_map().items():
        if spec['category'] == 'video' and spec['mode'] == mode:
            return type_id
    raise PackageError('Fast mode must be a pi-H3 package type.')


def _path(root):
    return Path(root) / FILE


def _lock(root):
    from .locking import runtime_lock
    return runtime_lock(Path(root) / LOCK_NAME, inherit=False, wait_seconds=3)


@contextmanager
def _engine_lease(root, message=BUSY_IMAGE):
    from .locking import runtime_lock
    try:
        with runtime_lock(Path(os.environ.get('FREEVIDEO_LOCK_PATH') or Path(root) / 'engine.lock'),
                          inherit=False, wait_seconds=0):
            yield
    except BlockingIOError:
        raise PackageError(message) from None


@contextmanager
def _video_lease(root):
    from .model_components import usage_lock
    message = 'Another task is using this model. Try again when it finishes.'
    try:
        with _engine_lease(root, message), usage_lock(root, wait_seconds=0):
            yield
    except BlockingIOError:
        raise PackageError(message) from None


def _matches(kind, value, default):
    if kind == 'int':
        return type(value) is int and value == default
    if kind == 'floats':
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != len(default):
            return False
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) and item == expected
                   for item, expected in zip(value, default))
    return False


def _clean_name(value, fallback):
    if value is None:
        value = fallback
    if not isinstance(value, str):
        raise PackageError('Enter a package name of 1 to 80 characters.')
    value = value.strip()
    if not 1 <= len(value) <= 80 or any(ord(character) < 32 for character in value):
        raise PackageError('Enter a package name of 1 to 80 characters.')
    return value


def _stored_parameters(spec, incoming, previous=None):
    """Keep only values that differ from a locked default. An equal locked value is dropped."""
    if not isinstance(incoming, dict):
        raise PackageError('Package parameters must be a set of named values.')
    known = {item['key']: item for item in spec['parameters']}
    if set(incoming) - set(known):
        raise PackageError('Unknown package parameter.')
    stored = {} if previous is None else dict(previous)
    for key, item in known.items():
        if key not in incoming:
            continue
        value = incoming[key]
        if not item['editable']:
            if not _matches(item['kind'], value, item['default']):
                raise PackageError('This parameter cannot be changed.')
            stored.pop(key, None)
            continue
        if item['kind'] == 'int' and (type(value) is not int
                                      or value < item.get('minimum', value) or value > item.get('maximum', value)):
            raise PackageError('This parameter is out of range.')
        stored[key] = value
    return stored


def _effective_parameters(spec, stored):
    values = {item['key']: item['default'] for item in spec['parameters']}
    for key, value in (stored or {}).items():
        values[key] = value
    return values


def _builtin_package(type_id):
    spec = type_map()[type_id]
    return dict(id=builtin_package_id(type_id), type=type_id, name=spec['label'], builtin=True,
                bindings={role: 'builtin' for role in spec['roles']}, parameters={})


def _user_package(type_id, name, bindings=None):
    spec = type_map()[type_id]
    package_id = 'pkg_' + secrets.token_hex(16)
    return dict(id=package_id, type=type_id, name=name, builtin=False,
                bindings=dict(bindings or {role: 'builtin' for role in spec['roles']}), parameters={})


def _remembered_family(root):
    from .image_gen.image_model import remembered
    try:
        configured = json.loads(io_path(Path(root) / 'machine.json').read_bytes())['model_root']
    except (OSError, ValueError, KeyError, TypeError):
        configured = None
    for folder in (configured, Path(root) / 'models' / 'vdn', Path(root) / 'models', root):
        if not isinstance(folder, (str, Path)) or not folder:
            continue
        found = remembered(Path(folder).expanduser())
        if found:
            return found
    return None


def _image_defaults(root, packages, active=None):
    """Preserve the old per-family image defaults, including families outside the current page."""
    from .image_components import read_selections
    choices = read_selections(root)
    defaults = {}
    for type_id, spec in type_map().items():
        if spec['category'] != 'image':
            continue
        family = spec['family']
        slots = choices.get(family) or {}
        bindings = {role: slots.get(role) or 'builtin' for role in IMAGE_ROLES}
        if active and packages[active]['type'] == type_id:
            defaults[family] = active
            continue
        if not any(slots.values()):
            defaults[family] = builtin_package_id(type_id)
            continue
        package = next((item for item in packages.values()
                        if item['type'] == type_id and item['bindings'] == bindings), None)
        if package is None:
            package = _user_package(type_id, 'Saved ' + spec['label'], bindings)
            packages[package['id']] = package
        defaults[family] = package['id']
    return defaults, choices


def migrate(root):
    """Take one copy of the selections made before model packages."""
    from .image_gen import image_model
    from .model_components import selections
    from .model_components_generation import PI_SLOTS

    packages = {builtin_package_id(type_id): _builtin_package(type_id) for type_id in type_map()}
    defaults, choices = _image_defaults(root, packages)
    family = _remembered_family(root) or next((model for model in image_model.MODELS
                                             if not packages[defaults[model]]['builtin']), image_model.H3)
    legacy = {}
    selected = selections(root)
    for mode, slot in PI_SLOTS.items():
        type_id = type_for_mode(mode)
        identity = selected.get(slot)
        if identity:
            package = _user_package(type_id, 'Saved ' + type_map()[type_id]['label'],
                                    {role: identity for role in FAST_ROLES})
            packages[package['id']] = package
            legacy[mode] = package['id']
        else:
            legacy[mode] = builtin_package_id(type_id)
    encoder = (choices.get(image_model.H3) or {}).get('text_encoder') or 'builtin'
    return dict(schema_version=VERSION, active=dict(image=defaults[family]), image_defaults=defaults, legacy_fast=legacy,
                shared_encoder=encoder, packages=packages)


def _package_record(package_id, value):
    from .hf_image_models import PROFILE_ID
    from .model_components import ID

    type_id = value.get('type') if isinstance(value, dict) else None
    spec = type_map().get(type_id) if isinstance(type_id, str) else None
    if (not isinstance(value, dict) or set(value) != PACKAGE_KEYS or spec is None or value.get('id') != package_id
            or not isinstance(value.get('name'), str) or not value['name'].strip() or len(value['name']) > 80
            or any(ord(character) < 32 for character in value['name'])
            or not isinstance(value.get('bindings'), dict) or set(value['bindings']) != set(spec['roles'])
            or not isinstance(value.get('parameters'), dict) or type(value.get('builtin')) is not bool):
        raise PackageError('Saved model packages could not be read.')
    builtin_id = builtin_package_id(value['type'])
    if value['builtin'] != (package_id == builtin_id) or (value['builtin'] and value['parameters']):
        raise PackageError('Saved model packages could not be read.')
    if not value['builtin'] and not PACKAGE_ID.fullmatch(package_id):
        raise PackageError('Saved model packages could not be read.')
    for role, binding in value['bindings'].items():
        if binding == 'builtin':
            continue
        if value['builtin'] or not isinstance(binding, str):
            raise PackageError('Saved model packages could not be read.')
        if spec['category'] == 'image':
            if not PROFILE_ID.fullmatch(binding):
                raise PackageError('Saved model packages could not be read.')
        elif not ID.fullmatch(binding):
            raise PackageError('Saved model packages could not be read.')
    _stored_parameters(spec, value['parameters'])
    if value['builtin'] and any(binding != 'builtin' for binding in value['bindings'].values()):
        raise PackageError('Saved model packages could not be read.')


def _document(value, *, version=VERSION):
    from .hf_image_models import PROFILE_ID

    keys = LEGACY_KEYS if version == 1 else TOP_KEYS
    if not isinstance(value, dict) or set(value) != keys or type(value.get('schema_version')) is not int or value['schema_version'] != version:
        raise PackageError('Saved model packages could not be read.')
    packages, active, legacy = value.get('packages'), value.get('active'), value.get('legacy_fast')
    shared = value.get('shared_encoder')
    if (not isinstance(packages, dict) or not isinstance(active, dict) or set(active) != {'image'}
            or not isinstance(legacy, dict) or set(legacy) != {'fast2', 'fast4'}
            or not isinstance(shared, str) or (shared != 'builtin' and not PROFILE_ID.fullmatch(shared))):
        raise PackageError('Saved model packages could not be read.')
    for type_id in type_map():
        if builtin_package_id(type_id) not in packages:
            raise PackageError('Saved model packages could not be read.')
    for package_id, package in packages.items():
        if not isinstance(package_id, str) or package_id != (package.get('id') if isinstance(package, dict) else None):
            raise PackageError('Saved model packages could not be read.')
        _package_record(package_id, package)
    image = active.get('image')
    if not isinstance(image, str) or image not in packages or type_map()[packages[image]['type']]['category'] != 'image':
        raise PackageError('Saved model packages could not be read.')
    for mode, package_id in legacy.items():
        if not isinstance(package_id, str) or package_id not in packages or type_map()[packages[package_id]['type']].get('mode') != mode:
            raise PackageError('Saved model packages could not be read.')
    if version == VERSION:
        from .image_gen.image_model import MODELS
        defaults = value['image_defaults']
        if not isinstance(defaults, dict) or defaults.keys() != set(MODELS):
            raise PackageError('Saved model packages could not be read.')
        for family, package_id in defaults.items():
            if (not isinstance(package_id, str) or package_id not in packages
                    or type_map()[packages[package_id]['type']].get('family') != family):
                raise PackageError('Saved model packages could not be read.')
        if defaults[type_map()[packages[image]['type']]['family']] != image:
            raise PackageError('Saved model packages could not be read.')
    return value


def _read(path):
    try:
        value = json.loads(io_path(path).read_bytes())
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as error:
        raise PackageError('Saved model packages could not be read.') from error
    version = 1 if isinstance(value, dict) and value.get('schema_version') == 1 else VERSION
    return _document(value, version=version)


def _upgrade(root, document):
    if document['schema_version'] == 1:
        defaults, _ = _image_defaults(root, document['packages'], document['active']['image'])
        document.update(schema_version=VERSION, image_defaults=defaults)
    return document


def _write(root, document):
    _document(document)
    save(_path(root), document, durable=True)


def load(root):
    """The package document, migrated once when the file is absent."""
    path = _path(root)
    if io_path(path).is_file():
        document = _read(path)
        if document['schema_version'] == VERSION:
            return document
    with _lock(root):
        if io_path(path).is_file():
            document = _read(path)
            if document['schema_version'] == VERSION:
                return document
            document = _upgrade(root, document)
        else:
            document = migrate(root)
        _write(root, document)
        return document


def _change(root, update):
    with _lock(root):
        path = _path(root)
        document = _upgrade(root, _read(path)) if io_path(path).is_file() else migrate(root)
        update(document)
        _write(root, document)
        return document


def _machine(root, machine):
    if machine:
        return dict(machine, root=str(root))
    return dict(root=str(root), model_root=str(Path(root) / 'models'))


def _image_binding(identity):
    from .hf_image_models import PROFILE_ID

    if identity in (None, '', 'builtin'):
        return 'builtin'
    if isinstance(identity, str) and PROFILE_ID.fullmatch(identity):
        return identity
    raise PackageError('Choose an installed image component.')


def _check_image_binding(root, machine, family, role, identity):
    binding = _image_binding(identity)
    if binding == 'builtin':
        return binding
    from .image_components import installed
    try:
        installed(root, binding, _machine(root, machine), model=family, role=role)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        message = str(error).strip()
        if message == 'This weight belongs to a different model component.':
            raise PackageError(message) from error
        raise PackageError('This image component is missing or changed.') from error
    return binding


def _fast_filename(mode, role):
    from .fast_tier import CONSTANTS, mode_spec
    if role == 'adapter':
        return mode_spec(mode)['adapter']
    if role == 'constants':
        return CONSTANTS
    raise PackageError('Unknown package role.')


def _check_fast_binding(root, mode, role, identity, *, layout=False, machine=None):
    from .model_components import ID, intact, receipt
    from .model_components_generation import PI_SLOTS

    if identity in (None, '', 'builtin'):
        return 'builtin'
    if not isinstance(identity, str) or not ID.fullmatch(identity):
        raise PackageError('Choose an installed pi-H3 version.')
    slot = PI_SLOTS[mode]
    filename = _fast_filename(mode, role)
    try:
        folder, value = receipt(root, slot, identity)
        if not intact(folder, value) or filename not in value['files']:
            raise ValueError('missing')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise PackageError('This pi-H3 version is missing or changed.') from error
    if layout:
        from . import fast_pack
        from .model_components_generation import _validate_fast
        pack = fast_pack.catalog()
        model_root = Path((machine or {}).get('model_root') or Path(root) / 'models')
        fast_root = fast_pack._model_folder(dict(machine or {}, model_root=str(model_root)))
        rows = {row['file'].rsplit('/', 1)[-1]: row for row in fast_pack.rows(pack, mode=mode) if row.get('engine')}
        defaults = {name: fast_pack._target(fast_root, row) for name, row in rows.items()}
        try:
            _validate_fast(folder, {filename: defaults[filename]}, rows)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise PackageError('The replacement does not match this pi-H3 pack.') from error
    return identity


def image_default(root, model, *, document=None):
    if document is None:
        document = load(root)
    family = type_map()[type_for_family(model)]['family']
    return document['packages'][document['image_defaults'][family]]


def image_selections(root, *, document=None):
    from .image_gen.image_model import MODELS

    if document is None:
        document = load(root)
    return {model: {role: identity for role, identity in image_default(root, model, document=document)['bindings'].items()
                    if identity != 'builtin'} for model in MODELS}


def note_image_selection(root, model, role, identity, *, expected=None, shared_expected=None):
    """Update the default image package from the existing component controls under their engine lease."""
    if role not in IMAGE_ROLES:
        raise PackageError('Choose a supported image component.')
    binding = _image_binding(identity)
    type_id = type_for_family(model)

    def apply(document):
        target = image_default(root, model, document=document)
        if expected is not None and target != expected:
            raise PackageError('The active components changed during verification. Try again.')
        if shared_expected is not None:
            if document['shared_encoder'] != shared_expected:
                raise PackageError('The shared encoder changed during verification. Try again.')
            document['shared_encoder'] = binding
        if target['bindings'][role] == binding:
            return
        if target['builtin']:
            created = _user_package(type_id, 'Saved ' + type_map()[type_id]['label'])
            document['packages'][created['id']] = created
            document['image_defaults'][model] = created['id']
            if document['active']['image'] == target['id']:
                document['active']['image'] = created['id']
            target = created
        target['bindings'][role] = binding

    _change(root, apply)


def shared_encoder_binding(root):
    """The H3 text encoder video reads, or None for the built-in weight."""
    value = load(root).get('shared_encoder')
    return None if value in (None, '', 'builtin') else value


def binding_references(root, category, *, document=None):
    if document is None:
        document = load(root)
    bound = {binding for package in document['packages'].values()
             if type_map()[package['type']]['category'] == category for binding in package['bindings'].values()
             if binding != 'builtin'}
    if category == 'image' and document['shared_encoder'] != 'builtin':
        bound.add(document['shared_encoder'])
    return bound


def image_ready(machine, model):
    from .hf_image_models import FAMILIES
    from .image_gen import image_model, image_pack

    if model in (image_model.H3, image_model.MOSS) and (sys.platform == 'darwin' or machine.get('device_backend') == 'mps'):
        return False
    try:
        return bool(image_pack.ready(machine['model_root'], group=FAMILIES[model][1]))
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def resolve_image(machine, package_id=None, model=None, *, document=None):
    """The weights and locked parameters of one image package.

    A missing or changed file raises. The built-in weight is not substituted.
    """
    from .image_components import resolve

    root = machine.get('root')
    if not root:
        raise PackageError('Choose an installation before using a model package.')
    if document is None:
        document = load(root)
    if package_id in (None, ''):
        package_id = document['active']['image'] if model is None else image_default(root, model, document=document)['id']
    if not valid_package_id(package_id):
        raise PackageError('Choose an installed model package.')
    package = document['packages'].get(package_id)
    if package is None:
        raise PackageError('Choose an installed model package.')
    spec = type_map()[package['type']]
    if spec['category'] != 'image':
        raise PackageError('Choose an image model package.')
    if model is not None and spec['family'] != model:
        raise PackageError('This package does not match the selected image model.')
    selections = {role: None if package['bindings'][role] == 'builtin' else package['bindings'][role]
                  for role in IMAGE_ROLES}
    resolved = resolve(machine, spec['family'], selections=selections)
    parameters = _effective_parameters(spec, package['parameters'])
    frozen = tuple(sorted((key, tuple(value) if isinstance(value, list) else value) for key, value in parameters.items()))
    return dict(resolved, package_id=package['id'], type=package['type'], parameters=parameters,
                identity=(package['id'], resolved['identity'], frozen))


def _bound_root(model_root, engine_root):
    """The installation that owns this model folder, or None when the folder is not one."""
    if engine_root is not None:
        return Path(engine_root)
    if model_root is None:
        return None
    path = Path(model_root)
    for parent in (path, *path.parents):
        if io_path(parent / 'machine.json').is_file():
            return parent
    return None


def _fast_package(model_root, mode, *, engine_root=None, package_id=None):
    from .fast_tier import mode_spec
    mode_spec(mode)
    root = _bound_root(model_root, engine_root)
    if root is None:
        if package_id not in (None, ''):
            raise PackageError('Choose an installation before using a model package.')
        return None, None
    document = load(root)
    if package_id in (None, ''):
        package_id = document['legacy_fast'][mode]
    if not valid_package_id(package_id):
        raise PackageError('Choose an installed model package.')
    package = document['packages'].get(package_id)
    if package is None or type_map()[package['type']].get('mode') != mode:
        raise PackageError('This package does not match the fast mode.')
    return root, package


def fast_default_bindings(root, mode, *, document=None):
    if document is None:
        document = load(root)
    return document['packages'][document['legacy_fast'][mode]]['bindings']


def note_fast_selection(root, mode, identity):
    """Apply old single-version Activate and Reset operations under the component lease."""
    type_id = type_for_mode(mode)
    builtin = builtin_package_id(type_id)
    bindings = dict.fromkeys(FAST_ROLES, identity or 'builtin')

    def apply(document):
        if identity is None:
            document['legacy_fast'][mode] = builtin
            return
        package = next((item for item in document['packages'].values()
                        if item['type'] == type_id and item['bindings'] == bindings), None)
        if package is None:
            package = _user_package(type_id, 'Saved ' + type_map()[type_id]['label'], bindings)
            document['packages'][package['id']] = package
        document['legacy_fast'][mode] = package['id']

    _change(root, apply)


def resolve_fast(model_root, mode, *, engine_root=None, package_id=None):
    """Verified paths and hashes from a single selected pi-H3 package."""
    from .adaln_assets import asset_path
    from . import fast_pack
    from .fast_tier import CONSTANTS, FOLDER, mode_spec
    from .fast_tier import CONSTANTS_FILE
    from .model_components import intact, receipt, root_for_model_dir, selected_identity
    from .model_components_generation import PI_SLOTS

    root, package = _fast_package(model_root, mode, engine_root=engine_root, package_id=package_id)
    spec = mode_spec(mode)
    if package is None:
        root = root_for_model_dir(model_root)
        binding = selected_identity(root, PI_SLOTS[mode]) or 'builtin'
        bindings = dict.fromkeys(FAST_ROLES, binding)
    else:
        bindings = package['bindings']
    pinned = (('adapter', spec['adapter'], spec['adapter_sha256']), ('constants', CONSTANTS, CONSTANTS_FILE[1]))
    names = {filename for role, filename, _ in pinned if bindings[role] == 'builtin'}
    official = fast_pack.default_files(model_root, mode=mode, names=names) if names else {}
    result = dict(package_id=package['id'] if package else None, bindings=dict(bindings), paths={})
    imported = {}
    for role, filename, checksum in pinned:
        binding = bindings[role]
        if binding == 'builtin':
            try:
                path = official[f'{FOLDER}/{filename}']
            except KeyError as error:
                raise PackageError('The pi-H3 %s file is missing from the built-in pack.' % role) from error
        else:
            if binding not in imported:
                folder, value = receipt(root, PI_SLOTS[mode], binding)
                if not intact(folder, value):
                    raise PackageError('The pi-H3 %s file is missing or changed.' % role)
                imported[binding] = (folder, value)
            folder, value = imported[binding]
            if filename not in value['files']:
                raise PackageError('The pi-H3 %s file is missing or changed.' % role)
            path = asset_path(folder, filename)
            checksum = value['files'][filename]['sha256']
        result['paths'][filename] = path
        result[role + '_sha256'] = checksum
    return result


def fast_files(model_root, mode='fast2', *, engine_root=None, package_id=None):
    return resolve_fast(model_root, mode, engine_root=engine_root, package_id=package_id)['paths']


def fast_hashes(model_root, mode, *, engine_root=None, package_id=None):
    result = resolve_fast(model_root, mode, engine_root=engine_root, package_id=package_id)
    return result['adapter_sha256'], result['constants_sha256']


def _image_choices(root, machine, spec):
    from .hf_image_models import PROFILE_ID, PROFILES
    from .image_components import installed

    choices = {role: [dict(id='builtin', label='Built-in', label_zh='内置', builtin=True)] for role in spec['roles']}
    folder = io_path(Path(root) / PROFILES)
    if not folder.is_dir():
        return choices
    probe = _machine(root, machine)
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or not PROFILE_ID.fullmatch(child.name):
            continue
        try:
            row = installed(root, child.name, probe, model=spec['family'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        role = row.get('component')
        if role in choices:
            choices[role].append(dict(id=row['id'], label=row.get('label') or row['id'][:8], builtin=False))
    return choices


def _fast_choices(root, mode):
    from .model_components import ID, intact, receipt
    from .model_components_generation import PI_SLOTS

    names = {role: _fast_filename(mode, role) for role in FAST_ROLES}
    choices = {role: [dict(id='builtin', label='Built-in', label_zh='内置', builtin=True)] for role in FAST_ROLES}
    slot = PI_SLOTS[mode]
    base = io_path(Path(root) / 'model-components' / slot)
    if not base.is_dir():
        return choices
    for child in sorted(p for p in base.iterdir() if p.is_dir() and ID.fullmatch(p.name)):
        try:
            folder, value = receipt(root, slot, child.name)
            if not intact(folder, value):
                raise ValueError('changed')
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        label = str(value.get('label') or child.name[:8])
        for role, filename in names.items():
            if filename in value['files']:
                choices[role].append(dict(id=child.name, label=label, builtin=False))
    return choices


def _official_present(root, mode, machine, names=None):
    try:
        from . import fast_pack
        model_root = Path((machine or {}).get('model_root') or Path(root) / 'models')
        fast_root = fast_pack._model_folder(dict(machine or {}, model_root=str(model_root)))
        return fast_pack.default_present(fast_root, mode=mode, names=names)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _video_ready(root, package, machine):
    from .model_components import intact, receipt
    from .model_components_generation import PI_SLOTS

    spec = type_map()[package['type']]
    mode = spec['mode']
    names = {_fast_filename(mode, role) for role, binding in package['bindings'].items() if binding == 'builtin'}
    if names and not _official_present(root, mode, machine, names=names):
        return False
    imported = {}
    for role, binding in package['bindings'].items():
        if binding == 'builtin':
            continue
        try:
            if binding not in imported:
                folder, value = receipt(root, PI_SLOTS[mode], binding)
                if not intact(folder, value):
                    return False
                imported[binding] = value['files']
            if _fast_filename(mode, role) not in imported[binding]:
                return False
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
    return True


def _public_package(package, *, ready=None):
    spec = type_map()[package['type']]
    row = dict(id=package['id'], type=package['type'], name=package['name'], builtin=package['builtin'],
               category=spec['category'], bindings=dict(package['bindings']),
               parameters=_effective_parameters(spec, package['parameters']))
    if spec['category'] == 'image':
        row['family'] = spec['family']
    else:
        row['mode'] = spec['mode']
    if ready is not None:
        row['ready'] = ready
    return row


def _role_payload(spec, role, choices, family=None):
    label, label_zh = ROLE_TEXT[role]
    text = (FORMATS.get(family) or {}).get(role) if family else None
    if spec['category'] == 'video' and role == 'adapter':
        text = ('pi-H3 ConvRot INT8 adapter, one .safetensors file.', 'pi-H3 ConvRot INT8 适配器，单个 .safetensors 文件。')
    elif spec['category'] == 'video':
        text = ('pi-H3 AdaLN constants, one .safetensors file.', 'pi-H3 AdaLN 常量，单个 .safetensors 文件。')
    return dict(id=role, label=label, label_zh=label_zh, format=(text or ('', ''))[0], format_zh=(text or ('', ''))[1],
                choices=choices.get(role, []))


def status(root, machine=None, *, video_only=False):
    """Types, packages and the choices each role can bind. Package actions themselves are not downloads."""
    from .fast_tier import setup_reason

    document = load(root)
    machine = _machine(root, machine)
    types = []
    ready = {}
    for type_id, spec in type_map().items():
        if spec['category'] == 'image':
            if video_only:
                continue
            choices = _image_choices(root, machine, spec)
            roles = [_role_payload(spec, role, choices, spec['family']) for role in spec['roles']]
            ready[spec['family']] = image_ready(machine, spec['family'])
        else:
            choices = {} if video_only else _fast_choices(root, spec['mode'])
            roles = [] if video_only else [_role_payload(spec, role, choices) for role in spec['roles']]
        parameters = [dict(key=item['key'], label=item['label'], label_zh=item['label_zh'], editable=item['editable'],
                           value=item['default']) for item in spec['parameters']]
        row = dict(id=type_id, category=spec['category'], label=spec['label'], label_zh=spec['label_zh'],
                   roles=roles, parameters=parameters)
        if spec['category'] == 'image':
            row['family'] = spec['family']
        else:
            row['mode'] = spec['mode']
            row['official_present'] = _official_present(root, spec['mode'], machine)
        types.append(row)
    packages = []
    reason = setup_reason(machine)
    for package in document['packages'].values():
        spec = type_map()[package['type']]
        if spec['category'] == 'image':
            if video_only:
                continue
            usable = ready[spec['family']] and _image_package_ready(machine, package, document)
            row = _public_package(package, ready=usable)
        else:
            row = _public_package(package, ready=not reason and _video_ready(root, package, machine))
            if reason:
                row['compatibility_reason'] = reason
        packages.append(row)
    result = dict(types=types, packages=packages, legacy_fast=dict(document['legacy_fast']))
    if not video_only:
        result.update(active=dict(document['active']), shared_encoder=document['shared_encoder'], job=None)
    return result


def _image_package_ready(machine, package, document):
    try:
        resolve_image(machine, package_id=package['id'], document=document)
        return True
    except (OSError, ValueError):
        return False


def image_catalog(machine, *, family_readiness=None, document=None):
    """Image packages for the picture page: id, family, name, and whether every role can be opened."""
    root = machine.get('root')
    if not root:
        return []
    if document is None:
        try:
            document = load(root)
        except (OSError, ValueError):
            return []
    active = document['active']['image']
    rows = []
    for type_id, spec in type_map().items():
        if spec['category'] != 'image':
            continue
        family = spec['family']
        ready = image_ready(machine, family) if family_readiness is None else family_readiness.get(family, False)
        for package in document['packages'].values():
            if package['type'] != type_id:
                continue
            components_ready = _image_package_ready(machine, package, document)
            rows.append(dict(id=package['id'], type=type_id, family=spec['family'], name=package['name'],
                             builtin=package['builtin'], ready=ready and components_ready,
                             components_ready=components_ready, active=package['id'] == active,
                             type_label=spec['label'], type_label_zh=spec['label_zh']))
    return rows


def create(root, type_id, name=None):
    spec = type_map().get(type_id) if isinstance(type_id, str) else None
    if spec is None:
        raise PackageError('Choose a supported model package type.')
    package = _user_package(type_id, _clean_name(name, spec['label']))

    def apply(document):
        document['packages'][package['id']] = package

    _change(root, apply)
    return package


def _get_package(document, package_id):
    if not valid_package_id(package_id):
        raise PackageError('Choose an installed model package.')
    try:
        return document['packages'][package_id]
    except KeyError:
        raise PackageError('Choose an installed model package.') from None


def update(root, package_id, *, name=None, bindings=None, parameters=None, machine=None, rename=False):
    original = _get_package(load(root), package_id)
    spec = type_map()[original['type']]

    def apply(document):
        package = _get_package(document, package_id)
        if package['builtin']:
            raise PackageError('The built-in package cannot be changed.')
        if rename:
            package['name'] = _clean_name(name, package['name'])
        if bindings is not None:
            if not isinstance(bindings, dict) or any(role not in spec['roles'] for role in bindings):
                raise PackageError('Unknown package role.')
            checked = {}
            for role, identity in bindings.items():
                if spec['category'] == 'image':
                    checked[role] = _check_image_binding(root, machine, spec['family'], role, identity)
                else:
                    checked[role] = _check_fast_binding(root, spec['mode'], role, identity, layout=True, machine=machine)
            package['bindings'].update(checked)
        if parameters is not None:
            package['parameters'] = _stored_parameters(spec, parameters, package['parameters'])

    lease = _video_lease if spec['category'] == 'video' else _engine_lease
    with lease(root) if bindings is not None else nullcontext():
        return _change(root, apply)['packages'][package_id]


def activate(root, package_id, *, machine=None):
    """Make one image package current. A task that holds the image lease keeps the previous package."""

    def apply(document):
        package = _get_package(document, package_id)
        spec = type_map()[package['type']]
        if spec['category'] != 'image':
            raise PackageError('Choose an image model package.')
        if not image_ready(_machine(root, machine), spec['family']):
            raise PackageError('Install the built-in image model for this family first.')
        for role, binding in package['bindings'].items():
            if binding != 'builtin':
                _check_image_binding(root, machine, spec['family'], role, binding)
        document['active']['image'] = package_id
        document['image_defaults'][spec['family']] = package_id

    with _engine_lease(root):
        return _change(root, apply)


def remove(root, package_id):
    original = _get_package(load(root), package_id)
    spec = type_map()[original['type']]

    def apply(document):
        package = _get_package(document, package_id)
        if package['builtin']:
            raise PackageError('The built-in package cannot be removed.')
        if spec['category'] == 'image':
            if document['active']['image'] == package_id:
                raise PackageError(ACTIVE_IMAGE)
            defaults, key = document['image_defaults'], spec['family']
        else:
            defaults, key = document['legacy_fast'], spec['mode']
        if defaults[key] == package_id:
            defaults[key] = builtin_package_id(package['type'])
        del document['packages'][package_id]

    lease = _video_lease if spec['category'] == 'video' else _engine_lease
    with lease(root):
        return _change(root, apply)


def set_shared_encoder(root, identity, machine=None):
    """The encoder video reads. An image package's own encoder is left as it is."""
    with _engine_lease(root):
        return commit_shared_encoder(root, identity, machine)


def commit_shared_encoder(root, identity, machine=None, *, expected=None):
    """Commit the tested encoder while the model manager holds the shared engine lease."""
    from .image_gen import image_model

    binding = _image_binding(identity)
    if binding != 'builtin':
        _check_image_binding(root, machine, image_model.H3, 'text_encoder', binding)

    def apply(document):
        if expected is not None and document['shared_encoder'] != expected:
            raise PackageError('The shared encoder changed during verification. Try again.')
        document['shared_encoder'] = binding

    _change(root, apply)
    return binding


def perform(root, value, machine=None, *, change_shared_encoder=None):
    if not isinstance(value, dict):
        raise PackageError('Invalid model package request.')
    action = value.get('action')
    if action == 'create':
        create(root, value.get('type'), value.get('name'))
    elif action == 'update':
        update(root, value.get('id'), name=value.get('name'), bindings=value.get('bindings'),
               parameters=value.get('parameters'), machine=machine, rename='name' in value)
    elif action == 'activate':
        if value.get('surface', 'image') != 'image':
            raise PackageError('Choose the image surface.')
        activate(root, value.get('id'), machine=machine)
    elif action == 'remove':
        remove(root, value.get('id'))
    elif action == 'set-shared-encoder':
        identity = value.get('id', value.get('encoder'))
        if not valid_image_encoder(identity):
            raise PackageError('Choose the built-in or an installed H3 text encoder.')
        if change_shared_encoder is None:
            set_shared_encoder(root, identity, machine)
        else:
            change_shared_encoder(identity)
    else:
        raise PackageError('Unknown model package action.')
    return status(root, machine)


def valid_image_encoder(value):
    from .hf_image_models import PROFILE_ID
    return isinstance(value, str) and (value == 'builtin' or PROFILE_ID.fullmatch(value) is not None)
