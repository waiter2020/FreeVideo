"""Inspect, copy, verify and activate model components using the managed engine."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import stat
from tempfile import TemporaryDirectory
import threading
from urllib.parse import unquote, urlsplit

from . import disk_space, hf_auth, hf_video_catalog as hub, local_model_copy, model_components as models
from .adaln_assets import asset_path
from .file_ops import publish
from .hf_model_jobs import HubModelJobs, PENDING_LIMIT
from .locking import runtime_lock
from .long_paths import io_path, plain, resolved
from .monitoring import save
from .network import hash_file
from .storage import fingerprint


def approved_path(value, libraries):
    if not isinstance(value, str) or not value.strip() or '\0' in value or len(value) > 4096:
        raise ValueError('Enter a model file or directory from one of your model libraries.')
    path = Path(plain(io_path(Path(value.strip()).expanduser()).absolute()))
    if Path(plain(resolved(path))) != path:
        raise ValueError('Choose the resolved path to the model file or directory.')
    if not any(path.is_relative_to(Path(plain(resolved(root)))) for root in libraries if root):
        raise ValueError('Choose a file in a configured model library, or add its directory in Settings first.')
    if not io_path(path).exists():
        raise ValueError('The selected model path does not exist.')
    return path


def _names(folder):
    base = io_path(folder)
    result = set()
    seen = 0
    for root, directories, files in os.walk(base, followlinks=False):
        seen += len(directories) + len(files)
        if seen > 4096:
            raise ValueError('Choose a narrower model folder.')
        for name in directories[:]:
            path = Path(root) / name
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & 0x400:
                directories.remove(name)
                files.append(name)
        result.update((Path(root) / name).relative_to(base).as_posix() for name in files)
        if root != str(base) and not directories and not files:
            result.add(Path(root).relative_to(base).as_posix() + '/')
    return result


def _layout(slot, names, single=None):
    if single is not None:
        if slot.layout is not None or len(slot.files) != 1:
            raise ValueError('Choose the folder containing all files for this model component.')
        target, = slot.files
        hub._safe_path(target)
        if Path(target).suffix.lower() != Path(single).suffix.lower():
            raise ValueError(f'This component needs a {Path(target).suffix} file.')
        return {target: single}
    paths = sorted(slot.layout(names) if slot.layout is not None else slot.files)
    if not paths or len(paths) > 512 or len(paths) != len(set(paths)):
        raise ValueError('The model has an unsupported file layout.')
    for path in paths:
        hub._safe_path(path)
        if path in (models.RECEIPT, 'request.json', 'verified-downloads.json', 'download-progress.json', 'download.log'):
            raise ValueError('The model contains a reserved file name.')
        if path not in names:
            raise ValueError('The model is missing a required file: ' + path)
    return {name: name for name in paths}


def _public(candidate):
    return {key: candidate[key] for key in ('id', 'label', 'source', 'repo', 'revision', 'license', 'bytes', 'builtin_files')
            if key in candidate} | {'files': [row['target'] for row in candidate['files']]}


def _hub_selection(value):
    reference = value.get('repo')
    path = value.get('path') or ''
    revision = value.get('revision') or None
    if isinstance(reference, str) and reference.strip().startswith(hub.HUB + '/'):
        url = urlsplit(reference.strip())
        parts = unquote(url.path.strip('/')).split('/')
        if len(parts) >= 5 and parts[2] in ('blob', 'resolve') and not url.query and not url.fragment:
            if path or revision and revision != parts[3]:
                raise ValueError('Use either a Hugging Face file URL or an explicit repository file path.')
            reference, revision, path = '/'.join(parts[:2]), parts[3], '/'.join(parts[4:])
    repo, requested, prefix = hub.parse_reference(reference, revision=revision)
    if prefix and path:
        path = hub._at(prefix, path)
    return repo, requested, hub._safe_path(path or prefix or '', empty=True)


def _known_default(slot):
    paths = [Path(path) for path in slot.files.values() if path is not None]
    stats = []
    for path in paths:
        try:
            if io_path(path).is_file():
                stats.append(io_path(path).stat().st_size)
        except OSError:
            continue
    value = dict(id=models.BUILTIN, label=slot.label, source=slot.source or 'FreeVideo',
                 bytes=slot.default_bytes or sum(stats), installed=bool(slot.files) and len(paths) == len(stats),
                 license=slot.license)
    if slot.default_status is not None:
        value.update(slot.default_status())
    if 'actions' not in value:
        value['actions'] = []
        if slot.supported and not slot.manager:
            if slot.install_default:
                value['actions'].append('repair' if value['installed'] else 'install')
            if value['installed'] and (slot.verify_default or (slot.validate and paths)):
                value['actions'].append('verify')
    return value


def manager_versions(slot, default, states):
    state = states.get(slot.manager, {})
    if slot.manager == 'image':
        component = state.get('components', {}).get(slot.model, {}).get(slot.component, {})
        installed = component.get('installed', [])
        selected = component.get('active')
        supported = bool(component.get('ready'))
    else:
        native = slot.component == 'comfy_h3'
        matches = lambda row: (row.get('backend') == 'comfy_h3') == native
        installed = [row for row in state.get('installed', []) if matches(row)]
        selected = state.get('active')
        if selected is not None and not matches(selected):
            selected = None
        supported = bool((state.get('comfy_h3_hf_available') or installed) if native else state.get('available'))
    active = selected['id'] if selected else None
    versions = []
    for row in installed:
        builtin = row.get('builtin') or row['id'] == 'official'
        valid = not row.get('damaged')
        actions = ['verify'] if valid else []
        if not builtin and row['id'] != active and valid:
            actions.append('activate')
        if row.get('repairable'):
            actions.append('repair')
        if row.get('removable') and row['id'] != active:
            actions.append('remove')
        public = {key: row[key] for key in ('id', 'label', 'source', 'bytes', 'repo', 'revision', 'license', 'damaged') if key in row}
        public.setdefault('source', row.get('source_kind', 'FreeVideo' if builtin else 'local'))
        public.update(installed=valid, actions=actions if supported else [])
        if builtin:
            default.update(public)
        else:
            versions.append(public)
    if selected and active != default['id'] and not any(version['id'] == active for version in versions):
        versions.append(dict(id=active, label=selected['label'], installed=False, damaged=True, actions=[]))
    return active, versions, supported


class ComponentJobs(HubModelJobs):
    media = 'component'

    def __init__(self, root, machine, *args, retained=None, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.machine = machine
        self.picked = set()
        self.retained = retained if retained is not None else lambda identity: False

    def slot(self, name):
        slot = self._catalog().get(name) if isinstance(name, str) else None
        if slot is None:
            raise models.ComponentError('Choose a model component.')
        return slot

    def _catalog(self):
        return models.catalog(self.root, self.machine)

    def inspect(self, slot, value, libraries, token, *, use_builtin_files=False):
        if not slot.supported or slot.manager or slot.validate is None:
            raise ValueError(slot.reason or 'This model uses its existing management controls.')
        token = hf_auth.validate(token or '')
        source = value.get('source')
        if source == 'local':
            value_path = value.get('path')
            with self.mutex:
                key = (slot.id, value_path) if isinstance(value_path, str) else None
                picked = key in self.picked
                if picked:
                    self.picked.discard(key)
            path = Path(value_path) if picked else approved_path(value_path, libraries)
            is_file = io_path(path).is_file()
            folder = path.parent if is_file else path
            builtin_files = []
            if is_file and slot.local_file_records is not None:
                if Path(plain(resolved(path))) != path:
                    raise ValueError('Choose the resolved path to the model file or directory.')
                with os.scandir(io_path(folder)) as entries:
                    names = {entry.name for entry in entries if entry.is_file(follow_symlinks=False)}
                files, builtin_files = slot.local_file_records(path, names, use_builtin_files=use_builtin_files)
            else:
                layout = _layout(slot, set() if is_file else _names(folder), path.name if is_file else None)
                files = [local_model_copy.source_record(folder / name, target, root=folder)
                         for target, name in layout.items()]
            candidate = dict(label=path.name, source='Local', files=files, license='', local_path=str(path))
            if builtin_files:
                candidate['builtin_files'] = builtin_files
        elif source == 'hf':
            repo, requested, path = _hub_selection(value)
            info, entries = hub._info(repo, requested, token)
            repo, revision = info['id'], info['sha']
            is_file = path in entries
            folder = path.rpartition('/')[0] if is_file else path
            group = folder + '/' if folder else ''
            names = {name[len(group):] for name in entries if name.startswith(group)}
            layout = _layout(slot, names, path.rsplit('/', 1)[-1] if is_file else None)
            files = [hub._row(repo, revision, entries, group + name, target) for target, name in layout.items()]
            candidate = dict(label=repo, source='Hugging Face', files=files, license=hub._license(info),
                             repo=repo, revision=revision, path=path)
        else:
            raise ValueError('Choose a local model or a Hugging Face repository.')
        candidate.update(id=secrets.token_hex(16), slot=slot.id, source_kind=source, bytes=sum(row['bytes'] for row in files))
        with self.mutex:
            self.pending[candidate['id']] = candidate
            while len(self.pending) > PENDING_LIMIT:
                self.pending.pop(next(iter(self.pending)))
        return _public(candidate)

    def status(self, *, local=False, native_picker=False, managers=None):
        from .model_components_generation import PI_SLOTS
        from .model_packages import binding_references, fast_default_bindings, load as load_packages

        active = models.selections(self.root)
        rows = []
        modes = {slot: mode for mode, slot in PI_SLOTS.items()}
        packages, protected, package_error = None, set(), None
        for slot in self._catalog().values():
            default = _known_default(slot)
            versions = list(default.pop('versions', []))
            active_id = active.get(slot.id, default.pop('active_id', models.BUILTIN))
            supported = slot.supported
            if slot.id in modes:
                try:
                    if packages is None and package_error is None:
                        packages = load_packages(self.root)
                        protected = binding_references(self.root, 'video', document=packages)
                    if packages is not None:
                        bindings = set(fast_default_bindings(self.root, modes[slot.id], document=packages).values())
                        if len(bindings) == 1:
                            active_id, = bindings
                        else:
                            active_id = models.BUILTIN
                except (OSError, ValueError) as error:
                    package_error = str(error)
                if package_error:
                    supported = False
            if slot.manager and managers is not None:
                active_id, versions, supported = manager_versions(slot, default, managers)
                actions = ['reset'] if supported else []
            else:
                folder = models.slot_directory(self.root, slot.id)
                if io_path(folder).is_dir():
                    for directory in sorted(io_path(folder).iterdir()):
                        if models.ID.fullmatch(directory.name):
                            versions.append(self._version_status(slot, directory.name, active_id,
                                                                 protected if slot.id in modes else ()))
                if active_id != models.BUILTIN and not any(row['id'] == active_id for row in versions):
                    versions.append(self._version_status(slot, active_id, active_id))
                actions = ['inspect', 'install', 'verify', 'activate', 'reset', 'repair', 'remove'] if supported and slot.validate else []
            rows.append(dict(id=slot.id, group=slot.group, label=slot.label, label_zh=slot.label_zh,
                             consumers=slot.consumers, format=slot.format, supported=supported,
                             reason=(package_error or slot.reason) if slot.id in modes else slot.reason,
                             default=default, active_id=active_id, versions=versions,
                             actions=actions, local_selection=slot.local_selection if 'inspect' in actions else (),
                             local_uses_builtin_files=bool(slot.local_file_records and default['installed']),
                             manager=slot.manager, manager_model=slot.model,
                             manager_component=slot.component))
        with self.mutex:
            job = dict(self.task)
        return dict(slots=rows, job=job, local_available=local, native_picker=native_picker)

    def _version_status(self, slot, identity, active, protected=()):
        try:
            folder, value = models.receipt(self.root, slot.id, identity)
            valid = models.intact(folder, value)
        except models.ComponentError:
            return dict(id=identity, label=identity, installed=False, damaged=True, actions=[])
        actions = ['verify'] if valid else ['repair']
        if identity != active and valid:
            actions.append('activate')
            if identity not in protected:
                actions.append('remove')
        return (dict(id=identity, installed=valid, damaged=not valid, actions=actions)
                | {key: value[key] for key in ('label', 'source', 'repo', 'revision', 'license', 'bytes') if key in value})

    def perform(self, slot, name, value, token):
        if not slot.supported or slot.manager or slot.validate is None:
            raise ValueError(slot.reason or 'Open the model’s existing controls to manage it.')
        if name == 'cancel':
            with self.mutex:
                if self.task.get('slot') == slot.id:
                    self.cancel()
            return
        identity = value.get('version_id')
        if name == 'install' and identity != models.BUILTIN:
            with self.mutex:
                candidate = self.pending.get(value.get('candidate_id')) if isinstance(value.get('candidate_id'), str) else None
            if candidate is None or candidate['slot'] != slot.id:
                raise ValueError('Inspect this model again before installing it.')
            phase = 'copying' if candidate['source_kind'] == 'local' else 'downloading'
            self._begin(slot, phase, self._install, slot, candidate, token, total=candidate['bytes'])
        elif name in ('install', 'verify', 'repair') and identity == models.BUILTIN:
            self._begin(slot, 'verifying' if name == 'verify' else 'downloading', self._default, slot, name,
                        identity=identity, total=slot.default_bytes, setup_lock=False)
        elif name in ('activate', 'reset', 'verify', 'remove', 'repair'):
            if name != 'reset':
                models.version_directory(self.root, slot.id, identity)
            self._begin(slot, 'verifying' if name == 'verify' else 'waiting', self._change, slot, name, identity, token,
                        identity=identity)
        else:
            raise ValueError('Unknown model component action.')

    def _begin(self, slot, phase, work, *args, identity=None, total=0, setup_lock=True):
        with self.mutex:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError('Wait for the current model action to finish.')
            self.stop = threading.Event()
            self.task = {}
            self._update(phase, id=identity, slot=slot.id, done_bytes=0, total_bytes=total)
            self.thread = threading.Thread(target=self._run, args=(work, args, setup_lock),
                                           name='freevideo-model-component', daemon=True)
            self.thread.start()

    def _run(self, work, args, setup_lock):
        try:
            lease = runtime_lock(self.root / 'setup.lock', inherit=False) if setup_lock else nullcontext()
            with lease:
                self._check()
                work(*args)
            self._update('ready')
        except InterruptedError:
            self._update('cancelled')
        except BlockingIOError:
            self._update('failed', error='Another FreeVideo installation task is running. Retry when it finishes.')
        except Exception as error:
            logging.getLogger(__name__).debug('Model component action failed', exc_info=True)
            message = str(error) or type(error).__name__
            self._update('failed', error=message[:700])

    def _install(self, slot, candidate, token, repair=None):
        parent = models.store(self.root)
        if disk_space.free_bytes(parent) < candidate['bytes']:
            raise ValueError('Insufficient disk space for this model version.')
        parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='import-', dir=io_path(parent)) as temporary:
            workspace = Path(plain(temporary))
            staging = workspace / 'files'
            staging.mkdir()
            if candidate['source_kind'] == 'local':
                checksums = {name: checksum for name, _, checksum in local_model_copy.copy_files(
                    staging, candidate['files'], lambda done: self._progress_bytes('copying', done, candidate['bytes']))}
            else:
                self._download(self.machine, staging, candidate, hf_auth.validate(token or ''), {})
                checksums = {row['target']: row.get('sha256') or self._hash(asset_path(staging, row['target']))
                             for row in candidate['files']}
                # The transfer also maintains its resumable staging files. Move only verified model files.
                downloaded = staging
                staging = workspace / 'verified'
                for name in checksums:
                    target = io_path(asset_path(staging, name))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    publish(asset_path(downloaded, name), target)
            files = {name: dict(sha256=checksum, stamp=fingerprint(asset_path(staging, name)))
                     for name, checksum in checksums.items()}
            identity = hashlib.sha256(json.dumps([slot.id, sorted(checksums.items())]).encode()).hexdigest()[:32]
            if repair is not None and repair != identity:
                raise ValueError('The source changed. Import it as a new model version.')
            destination = models.version_directory(self.root, slot.id, identity)
            if io_path(destination).exists() and repair is None:
                if not models.intact(*models.receipt(self.root, slot.id, identity)):
                    raise ValueError('This model is already registered and needs repair.')
                self._update('ready', id=identity)
                return
            self._test(slot, staging, workspace)
            record = {key: candidate[key] for key in ('source', 'source_kind', 'label', 'license', 'bytes', 'repo', 'revision',
                                                      'path', 'local_path', 'builtin_files') if key in candidate}
            save(staging / models.RECEIPT, record | dict(schema_version=1, id=identity, slot=slot.id, files=files), durable=True)
            self._check()
            io_path(destination.parent).mkdir(parents=True, exist_ok=True)
            if repair is None:
                publish(staging, destination)
            else:
                with self._idle():
                    _, previous = models.receipt(self.root, slot.id, identity)
                    retired = asset_path(parent, 'retired-' + secrets.token_hex(16))
                    publish(destination, retired)
                    try:
                        publish(staging, destination)
                    except BaseException:
                        publish(retired, destination)
                        raise
                    self._discard_unchanged(retired, previous)
            self._update('ready', id=identity, done_bytes=candidate['bytes'])

    def _hash(self, path):
        return hash_file(path, progress=lambda *_: self._check())

    def _progress_bytes(self, phase, done, total):
        self._check()
        self._update(phase, done_bytes=done, total_bytes=total)

    @contextmanager
    def _idle(self):
        self._update('waiting')
        with self._smoke_lease() as descriptor:
            while True:
                self._check()
                acquired = False
                try:
                    with models.usage_lock(self.root, wait_seconds=.25):
                        acquired = True
                        self._check()
                        yield descriptor
                        return
                except BlockingIOError:
                    if acquired:
                        raise

    def _test(self, slot, folder, workspace):
        with self._idle() as descriptor:
            self._update('testing')
            status = workspace / 'test-result.json'
            self._child(self.machine, workspace, 'test', ['-m', 'freevideo_engine.model_component_jobs', str(self.root),
                        slot.id, str(folder), str(status)], descriptor=descriptor, timeout=600,
                        python=self.machine.get('comfy_python'))

    def _default(self, slot, action):
        installer = slot.repair_default if action == 'repair' and slot.repair_default else slot.install_default
        if action in ('install', 'repair') and installer:
            installer(lambda done, total: self._progress_bytes('downloading', done, total), self._check)
        elif action == 'verify' and slot.verify_default:
            slot.verify_default()
        elif action == 'verify' and slot.files and all(path is not None for path in slot.files.values()):
            name, path = next(iter(slot.files.items()))
            folder = Path(path).parents[len(Path(name).parts) - 1]
            if not all(asset_path(folder, name) == Path(path) for name, path in slot.files.items()):
                raise ValueError('Use the pack’s integrity check to verify this built-in component.')
            with TemporaryDirectory(prefix='freevideo-model-test-') as workspace:
                self._test(slot, folder, Path(workspace))
        else:
            raise ValueError('This built-in model uses a separate installer.')

    def _change(self, slot, action, identity, token):
        if action == 'reset':
            with self._idle():
                self._select(slot, None)
            return
        folder, value = models.receipt(self.root, slot.id, identity)
        if action == 'repair':
            local = value['source_kind'] == 'local'
            request = dict(source=value['source_kind'], path=value['local_path'] if local else value['path'],
                           repo=value.get('repo'), revision=value.get('revision'))
            if local:
                with self.mutex:
                    self.picked.add((slot.id, request['path']))
            inspected = self.inspect(slot, request, (), token, use_builtin_files=bool(value.get('builtin_files')))
            with self.mutex:
                candidate = self.pending.pop(inspected['id'])
            self._install(slot, candidate, token, repair=identity)
        elif action == 'verify':
            self._update('verifying', done_bytes=0, total_bytes=value['bytes'])
            for name, info in value['files'].items():
                if self._hash(asset_path(folder, name)) != info['sha256']:
                    raise ValueError('Model file failed verification: ' + name)
            if not models.intact(folder, value):
                raise ValueError('Model files changed on disk. Repair the installed version.')
            self._update('verifying', done_bytes=value['bytes'])
        else:
            with self._idle():
                if not models.intact(folder, value):
                    raise ValueError('Model files changed on disk. Repair the installed version.')
                if action == 'activate':
                    self._select(slot, identity)
                else:
                    self._remove(slot, folder, value)

    def _select(self, slot, identity):
        from .model_components_generation import PI_SLOTS
        from .model_packages import note_fast_selection

        for mode, name in PI_SLOTS.items():
            if slot.id == name:
                note_fast_selection(self.root, mode, identity)
                return
        models.choose(self.root, slot.id, identity)

    def _remove(self, slot, folder, value):
        from .model_components_generation import PI_SLOTS
        from .model_packages import binding_references

        if slot.id in PI_SLOTS.values():
            if value['id'] in binding_references(self.root, 'video'):
                raise ValueError('Remove this version from every pi-H3 package before deleting it.')
        elif models.selected_identity(self.root, slot.id) == value['id']:
            raise ValueError('Switch to another version before removing the active model.')
        if self.retained(value['id']):
            raise ValueError('This model is still needed by a saved working checkpoint.')
        names = set(value['files']) | {models.RECEIPT}
        paths = [asset_path(folder, name) for name in names]
        for path in paths:
            local_model_copy._source(path, folder)
        if _names(folder) != names:
            raise ValueError('This version folder contains additional files. FreeVideo kept it intact.')
        retired = folder.with_name('removed-' + secrets.token_hex(16))
        if not retired.is_relative_to(models.store(self.root)):
            raise ValueError('The model folder is outside the managed library.')
        publish(folder, retired)
        self._unlink(retired, names)

    @staticmethod
    def _unlink(retired, names):
        for name in names:
            io_path(asset_path(retired, name)).unlink()
        folders = {parent for name in names for parent in (retired / name).parents if parent.is_relative_to(retired)}
        for path in sorted(folders, key=lambda parent: len(parent.parts), reverse=True):
            io_path(path).rmdir()

    @staticmethod
    def _discard_unchanged(folder, receipt):
        try:
            for name, row in receipt['files'].items():
                path = asset_path(folder, name)
                if io_path(path).exists() and fingerprint(local_model_copy._source(path, folder)) == row['stamp']:
                    io_path(path).unlink()
            # Keep any changed bytes and their receipt for recovery.
            for root, directories, files in os.walk(io_path(folder), topdown=False, followlinks=False):
                if not directories and not files:
                    Path(root).rmdir()
        except (OSError, ValueError):
            logging.getLogger(__name__).debug('Could not tidy retired model files', exc_info=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('slot')
    parser.add_argument('folder', type=Path)
    parser.add_argument('status', type=Path)
    args = parser.parse_args()
    try:
        machine = json.loads(io_path(args.root / 'machine.json').read_bytes())
        slot = models.catalog(args.root, machine)[args.slot]
        if not slot.supported or slot.validate is None:
            raise ValueError('This model component is unavailable on the current engine.')
        slot.validate(args.folder)
    except Exception as error:
        save(args.status, dict(done=0, error=(str(error) or type(error).__name__)[:700]))
        raise


if __name__ == '__main__':
    main()
