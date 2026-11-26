"""Install and select compatible denoisers for the native image families."""
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import threading
import time

from . import disk_space, image_components
from .adaln_assets import asset_path, file_hash
from .hf_model_jobs import HubModelJobs, public_candidate
from .image_gen import image_model, image_pack
from .long_paths import io_path
from .locking import runtime_lock
from .monitoring import load_ledger, save
from .storage import fingerprint


FAMILIES = {
    image_model.H3: ('MiniMax H3 Image', 'h3-image:' + image_pack.default_tier()),
    image_model.QWEN: ('Qwen-Image 2.1', 'image'),
    image_model.MOSS: ('MOSS-Image', image_pack.MOSS_GROUP),
}
SELECTION = 'hf-image-models.json'
PROFILES = 'prepared/huggingface-image'
PROFILE_ID = re.compile(r'[0-9a-f]{24}')
BUILTIN = 'official:'
SMOKE_TIMEOUT = 30 * 60


def _read(path):
    return json.loads(io_path(path).read_text(encoding='utf-8'))


def _choices(engine_root):
    try:
        value = _read(Path(engine_root) / SELECTION)
        active = value.get('active')
        if value.get('schema_version') == 1 and isinstance(active, dict):
            return {model: identity for model, identity in active.items()
                    if model in FAMILIES and isinstance(identity, str) and PROFILE_ID.fullmatch(identity)}
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _folder(engine_root, identity):
    if not isinstance(identity, str) or not PROFILE_ID.fullmatch(identity):
        raise ValueError('Invalid image model ID.')
    return asset_path(engine_root, PROFILES + '/' + identity)


def _profile(engine_root, identity):
    folder = _folder(engine_root, identity)
    record = _read(folder / 'profile.json')
    profile = record['model']
    if (record.get('schema_version') != 1 or not isinstance(profile, dict) or profile.get('id') != identity
            or profile.get('model') not in FAMILIES or not isinstance(record.get('stamps'), dict)
            or not record['stamps']):
        raise ValueError('The installed image model record is invalid.')
    return folder, record


def _stamp(path):
    return fingerprint(io_path(path))


def _verify_stamps(folder, record):
    for name, stamp in record['stamps'].items():
        if (name != 'LICENSE' and not name.startswith('image_gen/')) or _stamp(asset_path(folder, name)) != stamp:
            raise ValueError('The installed image model changed. Inspect and import it again.')


def active_profile(engine_root, model, *, gpu_uuid=None):
    """Resolve only locally verified profiles for the native worker."""
    identity = _choices(engine_root).get(model)
    if identity is None:
        return None
    try:
        folder, record = _profile(engine_root, identity)
        gpu_uuid = gpu_uuid or _read(Path(engine_root) / 'machine.json').get('gpu_uuid')
        if record['model']['model'] == model and record.get('gpu_uuid') == gpu_uuid:
            _verify_stamps(folder, record)
            return dict(record['model'], root=str(folder))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def _ready(machine, model):
    try:
        return image_pack.ready(machine['model_root'], group=FAMILIES[model][1])
    except (OSError, ValueError, RuntimeError):
        return False


class ImageModels(HubModelJobs):
    media = 'image'

    def __init__(self, *args, comfy_h3_smoke=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_h3_smoke = comfy_h3_smoke

    def _machine(self):
        try:
            value = _read(self.root / 'machine.json')
            if value.get('ready') and all(value.get(key) for key in ('python', 'comfy_python', 'model_root', 'gpu_uuid')):
                return dict(value, root=str(self.root))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        raise ValueError('Install the FreeVideo engine on Windows with a CUDA GPU first.')

    def status(self):
        with self.mutex:
            task = dict(self.task)
        try:
            machine = self._machine()
        except ValueError:
            return dict(available=False, families=[], active={}, installed=[], task=task)
        families, installed, active, components = [], [], {}, {}
        try:
            choices = image_components.read_selections(self.root)
            selection_error = None
        except image_components.ComponentError as error:
            choices, selection_error = {}, str(error)
        for model, (label, _) in FAMILIES.items():
            ready = bool(_ready(machine, model))
            families.append(dict(id=model, label=label, ready=ready))
            slots = components[model] = {}
            for role in image_components.ROLES:
                shared = model == image_model.H3 and role == 'text_encoder'
                slot_ready = ready or (shared and image_components.shared_ready(machine))
                if not slot_ready:
                    slots[role] = dict(ready=False, active=None, installed=[])
                    continue
                builtin = dict(id=image_components.builtin_id(model, role), model=model, component=role,
                               label=label if role == 'dit' else 'FreeVideo ' + image_components.LABELS[role],
                               format='FreeVideo', builtin=True)
                identity = choices.get(model, {}).get(role)
                selected = builtin
                if identity:
                    try:
                        selected = image_components.public(image_components.installed(self.root, identity, machine,
                                                                                     model=model, role=role))
                    except (OSError, ValueError, TypeError, KeyError, AttributeError):
                        selected = dict(id=identity, model=model, component=role, label='Unavailable component', damaged=True)
                slots[role] = dict(ready=True, active=selected, installed=[dict(builtin, active=not identity)], shared=shared)
                if role == 'dit':
                    active[model] = selected
                    installed.extend(slots[role]['installed'])
        for folder in sorted((self.root / PROFILES).glob('*')):
            if not PROFILE_ID.fullmatch(folder.name):
                continue
            try:
                row = image_components.public(image_components.installed(self.root, folder.name, machine))
                slot = components[row['model']][row['component']]
                row['active'] = choices.get(row['model'], {}).get(row['component']) == row['id']
                slot['installed'].append(row)
                if row['component'] == 'dit':
                    installed.append(row)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
        return dict(available=True, families=families, active=active, installed=installed, task=task,
                    components=components, selection_error=selection_error)

    def search(self, query, token=None):
        from .hf_video_catalog import search
        self._machine()
        return search(query, token=token)

    def _check_slot(self, machine, model, component):
        image_components.check(model, component)
        if model == image_model.H3 and component == 'text_encoder' and image_components.shared_ready(machine):
            return
        if not _ready(machine, model):
            raise ValueError('Install the built-in image model for this family first.')

    def inspect(self, reference, model, token=None, component='dit'):
        from . import hf_image_catalog, image_component_catalog
        machine = self._machine()
        self._check_slot(machine, model, component)
        if component == 'dit':
            candidate = hf_image_catalog.inspect(reference, self.root, machine, model, token=token)
        else:
            candidate = image_component_catalog.inspect(reference, self.root, machine, model, component, token=token)
        candidate.update(component=component, base_files=[])
        # An inspected revision remains pinned. Include the expected content and role in its storage ID.
        contents = sorted((row['target'], row.get('sha256') or row.get('git_blob'), row['bytes'])
                          for row in candidate['files'])
        candidate['id'] = image_components.digest([model, component, contents])[:24]
        return self._remember(candidate)

    def local_picker_kind(self, model, component='dit'):
        self._check_slot(self._machine(), model, component)
        if component == 'dit':
            return 'moss' if model == image_model.MOSS else 'image'
        return 'encoder_shards' if model == image_model.MOSS and component == 'text_encoder' else component

    def inspect_local(self, selection, model, component='dit'):
        from . import local_image_catalog, image_component_catalog
        machine = self._machine()
        self._check_slot(machine, model, component)
        if component == 'dit':
            candidate = local_image_catalog.inspect(selection, self.root, machine, model)
            candidate['required_bytes'] -= sum(row['bytes'] for row in candidate['base_files'])
        else:
            candidate = image_component_catalog.inspect_local(selection, self.root, machine, model, component)
        candidate.update(component=component, base_files=[])
        return self._remember_local(candidate)

    def _profile_folder(self, identity):
        return _folder(self.root, identity)

    def _complete_profile(self, folder, machine):
        try:
            return image_components.public(image_components.installed(self.root, folder.name, machine))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        return None

    def _already_active(self, identity):
        machine = self._machine()
        try:
            row = image_components.installed(self.root, identity, machine)
            return image_components.read_selections(self.root).get(row['model'], {}).get(row['component']) == identity
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False

    def _stage(self, candidate, token, *, folder=None):
        if 'component' in candidate:
            return self._stage_component(candidate, token, folder)
        from . import hf_image_catalog, local_image_catalog, local_models, provision
        from .network import retain_partial
        machine = self._machine()
        model = candidate['model']
        if not _ready(machine, model):
            raise ValueError('Install the built-in image model for this family first.')
        local = folder is not None
        if not local:
            folder = _folder(self.root, candidate['id'])
            folder.mkdir(parents=True, exist_ok=True)
            if self._complete_profile(folder, machine) is not None:
                self._update('ready', done_bytes=candidate['bytes'])
                return
            remaining, ledger = self._remaining(folder, candidate['files'])
        companions = [(row, image_pack.target(row, machine['model_root']),
                       asset_path(folder, 'image_gen/' + row['file'])) for row in candidate['base_files']]
        companion_ledger = load_ledger(folder / 'verified-companions.json')
        to_import = []
        for row, source, target in companions:
            self._check()
            if not provision.verified(target, row, companion_ledger, progress=lambda *_: self._check()):
                to_import.append((row, source, target))
        if not local:
            reserve = sum(row['bytes'] for row, _, _ in to_import)
            margin = 1 << 30
            if disk_space.free_bytes(folder) < remaining + reserve + margin:
                raise ValueError('Insufficient disk space for this image model.')
            self._download(machine, folder, candidate, token, ledger)
        copied = candidate['bytes']

        def copying(event):
            self._check()
            if local:
                self._update('copying', done_bytes=copied + event['done_bytes'])

        for row, source, target in to_import:
            self._check()
            if not provision.verified(source, row, companion_ledger, progress=lambda *_: self._check()):
                raise ValueError('A built-in image model component failed verification. Repair the pack first.')
            if target.exists():
                retain_partial(target, 'hash-rejected')
            # A hardlink changes the stock file's Windows identity and invalidates
            # its readiness receipt while a profile is being prepared.
            record = dict(source=str(source), digest=local_models.expected(row), bytes=row['bytes'],
                          fingerprint=fingerprint(source), method='copy')
            local_models.import_file(row, target, record, callback=copying)
            copied += row['bytes']
            companion_ledger[str(target)] = provision.file_identity(target, row)
        save(folder / 'verified-companions.json', companion_ledger)
        self._check()
        pack = folder / 'image_gen'
        if local:
            self._update('verifying')
        catalog = local_image_catalog if local else hf_image_catalog
        catalog.validate_import(candidate, pack, machine)
        self._update('waiting_for_gpu')
        with self._smoke_lease() as descriptor:
            self._update('smoke')
            output = folder / ('smoke-' + str(time.time_ns()) + '.png')
            self._child(machine, folder, 'smoke', ['-m', 'freevideo_engine.image_gen.variant_smoke',
                        '--pack', str(pack), '--machine', str(self.root / 'machine.json'), '--model', model,
                        '--out', str(output)], descriptor=descriptor, timeout=SMOKE_TIMEOUT, python=machine['comfy_python'])
            if not output.is_file() or output.stat().st_size < 32:
                raise ValueError('The candidate image model did not produce a picture.')
        self._check()
        paths = [row['target'] for row in candidate['files']] + ['image_gen/' + row['file'] for row in candidate['base_files']]
        stamps = {path: _stamp(asset_path(folder, path)) for path in paths}
        published = public_candidate(candidate)
        save(folder / 'profile.json', dict(schema_version=1, model=published, stamps=stamps,
                                          gpu_uuid=machine['gpu_uuid']), durable=True)
        if not local:
            self._update('ready')
        return published

    def _stage_component(self, candidate, token, folder):
        from . import hf_image_catalog, image_component_catalog, local_image_catalog, provision

        machine = self._machine()
        model, role = candidate['model'], candidate['component']
        self._check_slot(machine, model, role)
        local = folder is not None
        if not local:
            folder = _folder(self.root, candidate['id'])
            folder.mkdir(parents=True, exist_ok=True)
            if self._complete_profile(folder, machine) is not None:
                self._update('ready', done_bytes=candidate['bytes'])
                return
            remaining, ledger = self._remaining(folder, candidate['files'])
            if disk_space.free_bytes(folder) < remaining + hf_image_catalog.hub.MARGIN:
                raise ValueError('Insufficient disk space for this image component.')
            self._download(machine, folder, candidate, token, ledger)

        self._update('verifying')
        target = candidate['weights']
        if role == 'dit':
            catalog = local_image_catalog if local else hf_image_catalog
            catalog.validate_import(candidate, folder / 'image_gen', machine)
        else:
            image_component_catalog.validate_import(candidate, folder, machine)
        hashes = {name: checksum for name, _, checksum in candidate.get('local_hashes', [])}
        for row in candidate['files']:
            self._check()
            name = row['target']
            if name not in hashes:
                hashes[name] = row.get('sha256') or file_hash(io_path(asset_path(folder, name)))
        if model == image_model.MOSS:
            stock_dir = image_pack.base(machine['model_root']) / 'moss-image'
            stock_dir /= {'dit': 'transformer', 'text_encoder': 'text_encoder', 'vae': 'vae'}[role]
            config_name = 'config.json' if target == '.' else target + '/config.json'
            config = asset_path(folder, config_name)
            stock_file = stock_dir / 'config.json'
            manifest = next(row for row in image_pack.rows(group=FAMILIES[model][1])
                            if row['file'].endswith('/' + stock_dir.name + '/config.json'))
            if not provision.verified(stock_file, manifest, {}, progress=lambda *_: self._check()):
                raise ValueError('The built-in MOSS component configuration is invalid.')
            shutil.copyfile(io_path(stock_file), io_path(config))
            hashes[config_name] = manifest['sha256']
        stamps = {name: _stamp(asset_path(folder, name)) for name in hashes}
        self._check()
        published = public_candidate(candidate)
        save(folder / image_components.RECEIPT, dict(schema_version=1, model=published, weights=target,
                                                     hashes=hashes, stamps=stamps), durable=True)
        if not local:
            self._update('ready', candidate=published, done_bytes=candidate['bytes'])
        return published

    def start(self, identity, token=None, *, local=False, model=None, component=None):
        if model is not None or component is not None:
            with self.mutex:
                candidate = self.pending.get(identity) if isinstance(identity, str) else None
                if candidate and ((component is not None and candidate.get('component', 'dit') != component)
                                  or (model is not None and candidate['model'] != model)):
                    raise ValueError('Inspect this component before importing it.')
        return super().start(identity, token=token, local=local)

    def activate(self, identity, *, model=None, component=None):
        if not isinstance(identity, str):
            raise ValueError('Choose an installed image model.')
        if model is not None or component is not None or (not identity.startswith(BUILTIN)
                                                         and (_folder(self.root, identity) / image_components.RECEIPT).is_file()):
            return self._start_activation(identity, model, component)
        if identity.startswith(BUILTIN):
            if identity.removeprefix(BUILTIN) not in FAMILIES:
                raise ValueError('Unknown built-in image model.')
        else:
            _folder(self.root, identity)
        return self._switch_when_idle(identity)

    def _start_activation(self, identity, model, role):
        machine = self._machine()
        if identity.startswith(BUILTIN):
            parts = identity.split(':')
            model = model or parts[1]
            role = role or (parts[2] if len(parts) == 3 else 'dit')
            self._check_slot(machine, model, role)
            if identity != image_components.builtin_id(model, role):
                raise ValueError('The built-in component does not match this selection.')
            selected = dict(id=identity, model=model, component=role, label='FreeVideo ' + image_components.LABELS[role])
        else:
            selected = image_components.public(image_components.installed(self.root, identity, machine,
                                                                           model=model, role=role))
            model, role = selected['model'], selected['component']
            self._check_slot(machine, model, role)

        with self.mutex:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError('Wait for the image component operation to finish.')
            self.stop = threading.Event()
            self.task = {}
            self._update('waiting_for_gpu', id=identity, candidate=selected, operation='activate',
                         done_bytes=0, total_bytes=0)
            self.thread = threading.Thread(target=self._verify_activation, args=(selected,),
                                           name='freevideo-image-component-activation', daemon=True)
            self.thread.start()
        return self.status()

    def _verify_activation(self, selected):
        from .long_paths import plain

        model, role, identity = selected['model'], selected['component'], selected['id']
        builtin = identity == image_components.builtin_id(model, role)
        shared = model == image_model.H3 and role == 'text_encoder'
        try:
            with runtime_lock(self.root / 'setup.lock', inherit=False):
                machine = self._machine()
                choices = image_components.read_selections(self.root)
                proposed = {family: dict(slots) for family, slots in choices.items()}
                proposed.setdefault(model, {})[role] = None if builtin else identity
                if proposed == choices:
                    self._update('ready')
                    return
                encoder = image_components.shared_encoder(machine, identity=None if builtin else identity) if shared else None
                picture = image_components.resolve(machine, model, selections=proposed[model]) if _ready(machine, model) else None
                native = self._native_presets(machine, encoder) if shared else []
                vdn = shared and machine.get('video_model', True) and machine.get('cache')
                with TemporaryDirectory(prefix='component-activation-', dir=io_path(self.root)) as temporary:
                    folder = Path(plain(temporary))
                    self.release_idle()
                    if picture is not None or vdn:
                        with self._smoke_lease() as descriptor:
                            if picture is not None:
                                self._activate_picture(machine, folder, descriptor, picture)
                            if vdn:
                                from .hf_video_models import smoke_shared_encoder
                                self._update('smoke', validation='FreeVideo VDN')
                                for reference, phase in ((False, 'video-text'), (True, 'video-reference')):
                                    smoke_shared_encoder(self._child, machine, folder, descriptor,
                                                         None if builtin else identity, reference=reference, phase=phase)
                    for profile in native:
                        self._check()
                        self._update('smoke', validation=profile['preset'])
                        output = folder / (profile['id'] + '.mp4')
                        result = self.comfy_h3_smoke(profile, output, self._check)
                        self._child(machine, folder, 'validate-video-' + profile['preset'],
                                    ['-m', 'freevideo_engine.hf_video_models', 'validate-video', str(result)], timeout=30)
                    self._update('waiting_for_gpu', validation=None)
                    self.release_idle()
                    with self._smoke_lease():
                        self._check()
                        if image_components.read_selections(self.root) != choices:
                            raise ValueError('The active components changed during verification. Try again.')
                        if picture is not None:
                            latest = image_components.resolve(machine, model, selections=proposed[model])
                            if latest['identity'] != picture['identity']:
                                raise ValueError('A component changed during verification. Import it again.')
                        if shared:
                            image_components.shared_encoder(machine, identity=None if builtin else identity)
                        image_components.write_selections(self.root, proposed)
                self._update('ready', validation=None)
        except InterruptedError:
            self._update('cancelled', validation=None)
        except BlockingIOError:
            self._update('failed', validation=None, error='Another FreeVideo installation task is running. Try again.')
        except Exception as error:
            self._update('failed', error=(str(error) or type(error).__name__)[:700])

    def _activate_picture(self, machine, folder, descriptor, picture):
        model = picture['model']
        scenarios = {image_model.QWEN: ('text', 'reference', 'mask'), image_model.H3: ('text', 'reference'),
                     image_model.MOSS: ('text',)}[model]
        components = folder / 'components.json'
        save(components, picture)
        for scenario in scenarios:
            self._check()
            self._update('smoke', validation=scenario)
            output = folder / (scenario + '.png')
            self._child(machine, folder, 'image-' + scenario, ['-m', 'freevideo_engine.image_gen.variant_smoke',
                        '--pack', picture['pack'], '--machine', str(self.root / 'machine.json'), '--model', model,
                        '--components-json', str(components), '--scenario', scenario, '--out', str(output)],
                        descriptor=descriptor, timeout=SMOKE_TIMEOUT, python=machine['comfy_python'])
            if not output.is_file() or output.stat().st_size < 32:
                raise ValueError('The candidate image components did not produce a valid test picture.')

    def _native_presets(self, machine, encoder):
        from .hf_video_models import VideoModels, comfy_shared_encoder_profile

        models = VideoModels(self.root).status().get('installed', ())
        identities = [row['id'] for row in models if row.get('kind') == 'comfy_h3']
        if identities and (self.comfy_h3_smoke is None or not self.comfy_h3_smoke.available()):
            raise ValueError('Open the local ComfyUI queue to verify the shared encoder for installed H3 presets.')
        return [comfy_shared_encoder_profile(self.root, identity, machine, encoder) for identity in identities]

    def _activate(self, identity):
        machine = self._machine()
        saved = _choices(self.root)
        if identity.startswith(BUILTIN):
            saved.pop(identity.removeprefix(BUILTIN), None)
        else:
            try:
                folder, record = _profile(self.root, identity)
            except FileNotFoundError as error:
                raise ValueError('Install this image model before activating it.') from error
            model = record['model']['model']
            if record['gpu_uuid'] != machine['gpu_uuid']:
                raise ValueError('This model was tested on a different GPU. Import it again.')
            if not _ready(machine, model):
                raise ValueError('Install the built-in image model for this family first.')
            _verify_stamps(folder, record)
            saved[model] = identity
        save(self.root / SELECTION, dict(schema_version=1, active=saved), durable=True)
