"""Resolve installed image weights and the single shared H3 encoder selection."""
import hashlib
import json
from pathlib import Path

from .adaln_assets import asset_path
from .image_gen import image_model, image_pack
from .long_paths import io_path
from .monitoring import save
from .storage import fingerprint


ROLES = ('dit', 'text_encoder', 'vae')
LABELS = {'dit': 'DiT', 'text_encoder': 'Encoder', 'vae': 'VAE'}
SELECTION = 'image-components.json'
RECEIPT = 'component.json'
ACTIVE = object()


class ComponentError(ValueError):
    """A saved component could not be used. Never substitute another weight silently."""

    def __init__(self, message, role=None):
        super().__init__(message)
        self.role = role


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check(model, role):
    if model not in image_model.MODELS or role not in ROLES:
        raise ValueError('Choose a supported image model and component.')


def builtin_id(model, role):
    return 'official:' + model + (':' + role if role != 'dit' else '')


def read_selections(root):
    from .hf_image_models import _choices, PROFILE_ID

    choices = {model: {'dit': identity} for model, identity in _choices(root).items()}
    try:
        data = json.loads(io_path(Path(root) / SELECTION).read_bytes())
    except FileNotFoundError:
        return choices
    except (OSError, ValueError) as error:
        raise ComponentError('Saved image component selections could not be read.') from error
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('active'), dict):
        raise ComponentError('Saved image component selections are invalid.')
    for model, selections in data['active'].items():
        if (model not in image_model.MODELS or not isinstance(selections, dict)
                or any(role not in ROLES or identity is not None and
                       (not isinstance(identity, str) or not PROFILE_ID.fullmatch(identity))
                       for role, identity in selections.items())):
            raise ComponentError('Saved image component selections are invalid.')
        choices.setdefault(model, {}).update(selections)
    return choices


def write_selections(root, choices):
    save(Path(root) / SELECTION, dict(schema_version=1, active=choices), durable=True)


def _stock(machine, model, role):
    from .hf_image_models import FAMILIES
    from .image_component_catalog import stock_paths

    root = machine['model_root']
    if role == 'dit':
        rows = image_pack.rows(group=FAMILIES[model][1])
        if model == image_model.MOSS:
            path = image_pack.base(root) / 'moss-image' / 'transformer'
            weights = [row for row in rows if row['file'].startswith('moss-image/transformer/')]
        else:
            weights = [next(row for row in rows if row['component'] == 'dit')]
            path = image_pack.target(weights[0], root)
    else:
        path = stock_paths(machine, model, role)
        if model == image_model.H3 and role == 'text_encoder':
            # This path is owned by setup and shared with the video routes.
            weights = [row for row in image_pack.rows(group=FAMILIES[model][1]) if row.get('shared')]
        else:
            weights = [row for row in image_pack.rows(group=FAMILIES[model][1]) if row['component'] == role]
    hashes = [(row['file'], row.get('sha256')) for row in weights]
    checksum = hashes[0][1] if len(hashes) == 1 and hashes[0][1] else digest(hashes)
    return dict(id=builtin_id(model, role), model=model, component=role, label='FreeVideo ' + LABELS[role],
                format='FreeVideo', builtin=True, path=str(path), sha256=checksum)


def _verified_file(folder, name, stamp):
    path = asset_path(folder, name)
    if not path.is_file() or fingerprint(path) != stamp:
        raise ValueError('The component file changed.')
    return path


def installed(root, identity, machine, *, model=None, role=None):
    from .hf_image_models import _folder, _profile, device_identity, FAMILIES, recorded_device

    folder = _folder(root, identity)
    try:
        receipt = json.loads(io_path(folder / RECEIPT).read_bytes())
    except FileNotFoundError:
        _, legacy = _profile(root, identity)
        row = dict(legacy['model'], component='dit')
        if recorded_device(legacy) != device_identity(machine):
            raise ValueError('This component was installed on a different GPU.')
        actual_model = row['model']
        if actual_model == image_model.MOSS:
            target = 'image_gen/moss-image/transformer'
            names = [name for name in legacy['stamps'] if name.startswith(target + '/')]
        else:
            stock = next(entry['file'] for entry in image_pack.rows(group=FAMILIES[actual_model][1])
                         if entry['component'] == 'dit')
            target = 'image_gen/' + stock
            # Some early receipts contained only one explicitly recorded denoiser.
            if target not in legacy['stamps']:
                alternatives = [name for name in legacy['stamps'] if name.endswith('denoiser.safetensors')]
                if len(alternatives) == 1:
                    target = alternatives[0]
            names = [target]
        if not names:
            raise ValueError('Missing denoiser files in the legacy receipt.')
        for name in names:
            _verified_file(folder, name, legacy['stamps'][name])
        row.update(sha256=digest(['legacy', identity]), path=str(asset_path(folder, target)))
    else:
        row = dict(receipt['model'])
        stamps = receipt['stamps']
        hashes = receipt['hashes']
        if (receipt.get('schema_version') != 1 or row.get('id') != identity or not stamps
                or set(stamps) != set(hashes)
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values())):
            raise ValueError('The component receipt is invalid.')
        target = receipt['weights']
        if target != '.' and target not in stamps and not any(name.startswith(target + '/') for name in stamps):
            raise ValueError('The component receipt does not include its weights.')
        for name, stamp in stamps.items():
            _verified_file(folder, name, stamp)
        weights = {name: sha for name, sha in hashes.items() if name.endswith('.safetensors')}
        if not weights:
            raise ValueError('The component has no weights.')
        checksum = next(iter(weights.values())) if len(weights) == 1 else digest(weights)
        row.update(path=str(folder if target == '.' else asset_path(folder, target)), sha256=checksum)
    check(row['model'], row['component'])
    if (model is not None and model != row['model']) or (role is not None and role != row['component']):
        raise ValueError('This weight belongs to a different model component.')
    return row


def _selected(machine, model, role, identity):
    try:
        return installed(machine['root'], identity, machine, model=model, role=role)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        from .hf_image_models import FAMILIES
        raise ComponentError(f'{FAMILIES[model][0]} {LABELS[role]} is missing or changed. '
                             'Import it again or restore the built-in component in Settings.', role=role) from error


def shared_encoder(machine, *, identity=ACTIVE):
    """Return the selected H3 encoder, or None for the installation's built-in one."""
    if identity is ACTIVE:
        root = machine.get('root')
        identity = read_selections(root).get(image_model.H3, {}).get('text_encoder') if root else None
    if identity in (None, '', 'builtin', builtin_id(image_model.H3, 'text_encoder')):
        return None
    return _selected(machine, image_model.H3, 'text_encoder', identity)


def shared_ready(machine):
    if not machine.get('encoder') or not machine.get('encoder_model_root'):
        return False
    path = Path(machine['encoder_model_root']) / 'text_encoders' / machine['encoder']
    return path.is_file()


def resolve(machine, model, *, selections=None):
    if selections is None:
        root = machine.get('root')
        selections = read_selections(root).get(model, {}) if root else {}
    components = {}
    for role in ROLES:
        identity = selections.get(role)
        components[role] = (_selected(machine, model, role, identity) if identity else _stock(machine, model, role))
    return dict(model=model, pack=str(image_pack.base(machine['model_root'])), components=components,
                identity=tuple((row['id'], row['sha256']) for row in components.values()))


def public(row):
    return {key: row[key] for key in ('id', 'model', 'component', 'label', 'format', 'repo', 'revision',
                                     'source_kind', 'bytes', 'sha256', 'builtin', 'damaged') if key in row}
