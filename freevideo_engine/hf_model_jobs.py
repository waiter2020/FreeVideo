"""Shared job lifecycle for staging and activating compatible model weights."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from tempfile import mkdtemp, TemporaryDirectory
import threading
import time

from . import hf_auth, processes
from .adaln_assets import asset_path
from .locking import LOCK_ENV, runtime_lock
from .long_paths import io_path, plain
from .monitoring import load_ledger, save


PENDING_LIMIT = 24
TASK_FIELDS = ('id', 'model', 'label', 'repo', 'revision', 'format', 'kind', 'subfolder', 'path', 'bytes',
               'download_bytes', 'source_kind', 'copy_bytes', 'required_bytes', 'file_count', 'license', 'license_url', 'compatible',
               'compatibility_reason', 'variants', 'backend', 'preset', 'component')


def public_candidate(candidate):
    value = {key: candidate[key] for key in TASK_FIELDS if key in candidate}
    if 'files' in candidate:
        value['file_count'] = len(candidate['files'])
    return value


class HubModelJobs:
    """Serialize downloads, GPU validation, and activation for one model selector."""

    def __init__(self, root, source=None, queue_busy=None, about_to_switch=None, release_idle=None):
        self.root = Path(root).resolve()
        self.source = Path(source or Path(__file__).resolve().parents[1])
        self.queue_busy = queue_busy or (lambda: False)
        self.about_to_switch = about_to_switch or (lambda: None)
        self.release_idle = release_idle or (lambda: None)
        self.mutex = threading.RLock()
        self.pending = {}
        self.thread = None
        self.stop = threading.Event()
        self.task = dict(phase='idle', status='idle', done=0, total=0, done_bytes=0, total_bytes=0)

    def _update(self, phase, **fields):
        with self.mutex:
            self.task.update(phase=phase, status=phase, **fields)
            self.task['done'] = self.task['done_bytes']
            self.task['total'] = self.task['total_bytes']

    def _remember(self, candidate):
        with self.mutex:
            self.pending[candidate['id']] = candidate
            while len(self.pending) > PENDING_LIMIT:
                self.pending.pop(next(iter(self.pending)))
        return public_candidate(candidate)


    def _remember_local(self, candidate):
        return self._remember(dict(candidate, id=secrets.token_hex(12)))

    def start(self, identity, token=None, *, local=False):
        if not isinstance(identity, str):
            raise ValueError('Choose an inspected model.')
        with self.mutex:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError(f'A {self.media} model import is already running.')
            if self._already_active(identity):
                raise ValueError(f'This {self.media} model is already active.')
            candidate = self.pending.get(identity)
            if candidate is None or (candidate.get('source_kind') == 'local') != local:
                raise ValueError('Select the local model again before copying.' if local
                                 else 'Inspect the Hugging Face model again before downloading.')
            if not candidate.get('compatible', True):
                raise ValueError(candidate.get('compatibility_reason') or 'The model is incompatible.')
            token = '' if local else hf_auth.validate(token or '')
            self.stop = threading.Event()
            self.task = {}
            fields = {'selection_id': identity} if local else {}
            self._update('copying' if local else 'downloading', id=identity, candidate=public_candidate(candidate), done_bytes=0,
                         total_bytes=candidate['copy_bytes'] if local else candidate['bytes'], **fields)
            self.thread = threading.Thread(target=self._work, args=(candidate, token),
                                           name='freevideo-' + self.media + '-model-import', daemon=True)
            self.thread.start()
            return dict(self.task)

    def cancel(self):
        with self.mutex:
            self.stop.set()
            return dict(self.task)

    def _maintenance(self, identity, action, work, *args, token='', lock=True):
        with self.mutex:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError(f'Wait for the current {self.media} model action to finish.')
            self.stop = threading.Event()
            self.task = {}
            phase = {'verify': 'verifying', 'repair': 'verifying', 'remove': 'waiting_for_gpu'}[action]
            self._update(phase, id=identity, operation=action, done_bytes=0, total_bytes=0)
            self.thread = threading.Thread(target=self._run_maintenance, args=(work, args, token, lock),
                                           name='freevideo-' + self.media + '-model-' + action, daemon=True)
            self.thread.start()
            return dict(self.task)

    def _run_maintenance(self, work, args, token, lock):
        try:
            with runtime_lock(self.root / 'setup.lock', inherit=False) if lock else nullcontext():
                self._check()
                work(*args)
            self._update('ready')
        except InterruptedError:
            self._update('cancelled')
        except BlockingIOError:
            self._update('failed', error='Another FreeVideo installation task is running. Retry when it finishes.')
        except Exception as error:
            message = str(error).replace(token, '[redacted]') if token else str(error)
            self._update('failed', error=(message or type(error).__name__)[:700])

    @contextmanager
    def _maintenance_lease(self):
        self._update('waiting_for_gpu')
        self.release_idle()
        with self._smoke_lease():
            self._check()
            yield

    def _delete_import(self, folder):
        from .file_ops import publish
        folder = io_path(folder)
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError('The model is no longer installed.')
        with TemporaryDirectory(prefix='removed-model-', dir=folder.parent) as temporary:
            destination = Path(temporary) / 'model'
            publish(folder, destination)
            shutil.rmtree(destination)

    def _replace_import(self, folder, staged, update):
        from .file_ops import publish
        folder = io_path(folder)
        temporary = Path(mkdtemp(prefix='replace-model-', dir=folder.parent))
        backup = temporary / 'original'
        try:
            publish(folder, backup)
            try:
                publish(staged, folder)
                update()
            except BaseException:
                if folder.is_dir():
                    publish(folder, staged)
                publish(backup, folder)
                raise
        finally:
            if not backup.exists():
                temporary.rmdir()
        shutil.rmtree(temporary)

    def _check(self):
        if self.stop.is_set():
            raise InterruptedError(f'{self.media.capitalize()} model import cancelled.')

    def _engine_lock(self):
        return Path(os.environ.get('FREEVIDEO_LOCK_PATH') or self.root / 'engine.lock')

    @contextmanager
    def _smoke_lease(self):
        while True:
            self._check()
            if self.queue_busy() is False:
                acquired = False
                try:
                    with runtime_lock(self._engine_lock(), inherit=False) as descriptor:
                        acquired = True
                        if self.queue_busy() is False:
                            self.about_to_switch()
                            if self.queue_busy() is False:
                                yield descriptor
                                return
                except BlockingIOError:
                    if acquired:
                        raise
                    # Idle prewarming can hold the lease even with an empty queue.
                    if self.queue_busy() is False:
                        self.release_idle()
            self.stop.wait(.5)

    def _work(self, candidate, token):
        try:
            with runtime_lock(self.root / 'setup.lock', inherit=False):
                self._check()
                if candidate.get('source_kind') == 'local':
                    self._work_local(candidate)
                else:
                    self._stage(candidate, token)
        except InterruptedError:
            self._update('cancelled')
        except BlockingIOError:
            self._update('failed', error='Another FreeVideo installation task is running. Retry when it finishes.')
        except Exception as error:
            message = str(error).replace(token, '[redacted]') if token else str(error)
            if candidate.get('source_kind') == 'local':
                for row in (*candidate['files'], *candidate.get('dependencies', {}).values()):
                    names = sorted({row['source'], str(io_path(row['source'])), Path(row['source']).as_posix()},
                                   key=len, reverse=True)
                    for name in names:
                        message = message.replace(name, Path(row['source']).name)
            self._update('failed', error=(message or type(error).__name__)[:700])

    def _work_local(self, selection):
        from . import disk_space, local_model_copy
        from .file_ops import publish

        parent = asset_path(self.root, 'prepared/local-staging')
        parent.mkdir(parents=True, exist_ok=True)
        if disk_space.free_bytes(parent) < selection['required_bytes']:
            raise ValueError(f'Insufficient disk space to copy this {self.media} model.')

        def progress(done):
            self._check()
            self._update('copying', done_bytes=done)

        with TemporaryDirectory(prefix=self.media + '-', dir=io_path(parent)) as directory:
            staging = Path(plain(directory))
            hashes = self._copy_local_files(staging, selection, progress)
            self._check()
            attempt = 0
            while True:
                identity = local_model_copy.content_id(self.media, selection, hashes, attempt=attempt)
                destination = self._profile_folder(identity)
                installed = self._complete_profile(destination, self._machine())
                if installed is not None or not os.path.lexists(io_path(destination)):
                    break
                attempt += 1
            candidate = dict(selection, id=identity, local_hashes=hashes)
            if installed is None:
                installed = self._stage(candidate, '', folder=staging)
                self._check()
                destination.parent.mkdir(parents=True, exist_ok=True)
                publish(staging, destination)
            self._update('ready', id=identity, candidate=installed, done_bytes=selection['copy_bytes'])

    def _copy_local_files(self, staging, selection, progress):
        from .local_model_copy import copy_files
        return copy_files(staging, selection['files'], progress)

    def _child(self, machine, folder, name, args, *, token='', progress=None, descriptor=None, timeout=None, python=None):
        from .resident_process import ENV
        from .testing import environment
        env = environment(machine)
        env.pop(ENV, None)
        env.pop(LOCK_ENV, None)
        env['FREEVIDEO_LOCK_PATH'] = str(self._engine_lock())
        pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = str(self.source) + (os.pathsep + pythonpath if pythonpath else '')
        if token:
            env['HF_TOKEN'] = token
        if descriptor is not None:
            env[LOCK_ENV] = str(descriptor)
        start = time.monotonic()
        with (folder / (name + '.log')).open('wb') as log:
            process = processes.popen([python or machine['python'], '-B', *args], env=env, cwd=self.source,
                                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=True, supervise=True,
                                      pass_fds=() if descriptor is None else (descriptor,))
            try:
                while process.poll() is None:
                    self._check()
                    if timeout is not None and time.monotonic() - start >= timeout:
                        raise TimeoutError(f'{self.media.capitalize()} model {name} timed out.')
                    self._progress(progress)
                    self.stop.wait(.25)
                self._check()
                latest = self._progress(progress)
                if process.returncode:
                    raise RuntimeError(latest.get('error') or f'{self.media.capitalize()} model {name} failed. See {name}.log.')
            finally:
                if process.poll() is None:
                    processes.stop(process)

    def _progress(self, path):
        if path is not None:
            try:
                value = json.loads(io_path(path).read_text(encoding='utf-8'))
                self._update('downloading', done_bytes=value['done'])
                return value
            except (OSError, ValueError, TypeError, KeyError):
                pass
        return {}

    def _remaining(self, folder, rows):
        from . import provision
        ledger = load_ledger(folder / 'verified-downloads.json')
        remaining = 0
        for row in rows:
            self._check()
            if not provision.verified(asset_path(folder, row['target']), row, ledger,
                                      progress=lambda *_: self._check()):
                remaining += row['bytes']
        return remaining, ledger

    def _download(self, machine, folder, candidate, token, ledger):
        from .download_settings import read as read_preferences
        self._check()
        request = folder / 'request.json'
        status_path = folder / 'download-progress.json'
        status_path.unlink(missing_ok=True)
        preferences = read_preferences(self.root / 'download-settings.json')
        save(request, dict(files=candidate['files'], status=status_path.name,
                           proxy_mode=preferences['proxy_mode']), durable=True)
        save(folder / 'verified-downloads.json', ledger)
        self._child(machine, folder, 'download', ['-m', 'freevideo_engine.hf_video_models', 'transfer', str(request)],
                    token=token, progress=status_path)
        self._check()
        self._update('verifying', done_bytes=candidate['bytes'])

    def _switch_when_idle(self, identity):
        with self.mutex:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError(f'Wait for the {self.media} model import to finish.')
            if self.queue_busy() is not False:
                raise ValueError('Wait for the generation queue to finish.')
            try:
                with runtime_lock(self.root / 'setup.lock', inherit=False):
                    if self.queue_busy() is not False:
                        raise ValueError('Wait for the generation queue to finish.')
                    self.release_idle()
                    with runtime_lock(self._engine_lock(), inherit=False):
                        if self.queue_busy() is not False:
                            raise ValueError('Wait for the generation queue to finish.')
                        self.about_to_switch()
                        if self.queue_busy() is not False:
                            raise ValueError('Wait for the generation queue to finish.')
                        self._activate(identity)
            except BlockingIOError as error:
                raise ValueError('FreeVideo is busy. Retry when the current task finishes.') from error
        return self.status()
