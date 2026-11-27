"""On-demand downloads for the pi-H3 DWD variants and their shared timestep constants.

The catalog pins each file's size and SHA-256. A row can belong to particular modes
or be shared. Transfers reuse the image pack's source ordering, proxy settings,
resumable partials and integrity checks.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
from uuid import uuid4

from . import disk_space
from .locking import runtime_lock, windows_shared_lock
from .long_paths import io_path
from .monitoring import load_ledger, save
from .swap_lora import PackError

RESERVE = 256 * 2**20
MISSING = 'FAST_MODE_FILES_MISSING: '
DEFAULT_MODE = 'fast2'
IDLE = dict(state='idle', done=0, total=0, reason=None, job_id=None, cancelling=False)
JOB_LEDGER = 'download-job.json'
DOWNLOAD_LOCK = 'fast-pack-download.lock'
REPORT_INTERVAL = .25
# The receipt beside the files: what each one was when its sha256 last matched (size, times, file id). A file whose identity
# is unchanged is not hashed again; one that changed is checked again before it counts.
LEDGER = 'download-verified.json'
# A copy that did not pass, as network.retain_partial and its rejections name it: <file>[.partial].<reason>-<time_ns>,
# with the reasons they use and nothing else, so a name such as weights.bin.backup-1 is never taken for one.
FAILED = re.compile(r'(?:\.partial)?\.(?:rejected|hash-rejected|oversized|oversized-response|resume-unavailable|restart-approved)'
                    r'-[0-9]{16,20}')


def catalog():
    return json.loads(Path(__file__).with_name('fast_pack.json').read_text(encoding='utf-8'))


def _mode(mode):
    from .fast_tier import MODES
    if mode not in MODES:
        raise ValueError('Unsupported Fast mode')
    return mode


def rows(spec=None, *, mode=DEFAULT_MODE):
    """The shared files and the selected adapter, with licence and notes first."""
    _mode(mode)
    return [row for row in (spec or catalog())['files'] if 'modes' not in row or mode in row['modes']]


def _installation():
    from .comfy_bridge import installation
    try:
        return installation()
    except (OSError, ValueError, KeyError):
        return None, None


def _model_folder(machine):
    from .paths import installed_model_root
    override = os.environ.get('FREEVIDEO_MODEL_ROOT')
    if override:
        return Path(override).expanduser()
    return Path(machine['model_root']).expanduser() if machine else installed_model_root()


def model_folder():
    return _model_folder(_installation()[1])


def setup_compatibility(machine):
    """Report known device and engine limits without assuming a missing setup is unsupported."""
    from .fast_tier import setup_reason
    reason = setup_reason(machine)
    return dict(compatible=False if reason else True if machine else None, compatibility_reason=reason)


def remote(spec, row):
    """The pinned Hugging Face revision and mirrors for a catalog file."""
    from .network import source_url
    repo, revision = row.get('repo', spec['repo']), row.get('revision', spec['revision'])
    if revision is None:
        raise ValueError('Not on our copies yet: ' + row['file'])
    # A row from another repository has a ModelScope copy only if it names one.
    copy = row.get('modelscope') if 'repo' in row else row.get('modelscope', spec['modelscope'])
    path = '/%s/resolve/%s/%s' % (repo, revision, row['file'])
    sources = [dict(id='modelscope', url=copy + '/' + row['file'])] if copy else []
    sources.append(dict(id='official', url=source_url('models', 'official') + path))
    sources.append(dict(id='hf-mirror', url=row.get('hf_mirror') or source_url('models', 'hf-mirror') + path))
    return dict(repo=repo, revision=revision, file=row['file'], family='image',
                sha256=row['sha256'], bytes=row['bytes'], sources=sources)


def _target(root, row):
    """Where a row's file lives under the model folder; a catalog row never leaves it."""
    name = PurePosixPath(row.get('target', row['file']))
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Invalid catalog path')
    return Path(root) / name


def _ours(spec, row):
    """Whether the file lies in this pack's own folder; a file elsewhere belongs to the rest of the installation."""
    return PurePosixPath(row.get('target', row['file'])).parts[0] == spec['folder']


def _in_place(root, row):
    path = _target(root, row)
    return path.is_file() and path.stat().st_size == row['bytes']


def published(spec=None, *, mode=DEFAULT_MODE):
    """Whether every file is on our copies: a page that sees False does not offer the level at all."""
    spec = spec or catalog()
    return all(row.get('revision', spec['revision']) is not None for row in rows(spec, mode=mode))


def _ledger_path(root, spec):
    return Path(root) / spec['folder'] / LEDGER


def _ledger(root, spec, *, retain=False):
    """The receipt; a damaged one reads as empty. Only an install renames a damaged one aside (``retain``): reading the
    state leaves the folder as it is."""
    return load_ledger(_ledger_path(root, spec), retain=retain)


def _identity(path, row):
    from .storage import fingerprint
    return dict(fingerprint(path), sha256=row['sha256'])


def _receipted(root, row, ledger):
    """There at its size, and the same file the receipt says passed its sha256."""
    path = _target(root, row)
    try:
        return _in_place(root, row) and ledger.get(row['file']) == _identity(path, row)
    except (OSError, ValueError):
        return False


def _receive(root, spec, ledger, row, identity=None):
    """Note a file that has just passed its sha256."""
    try:
        stamp = identity if identity is not None else _identity(_target(root, row), row)
        ledger[row['file']] = stamp
        path = _ledger_path(root, spec)
        # A direct request may adopt one mode while the installer finishes the other.
        with runtime_lock(path.with_suffix('.lock'), inherit=False, wait_seconds=5):
            latest = _ledger(root, spec)
            latest[row['file']] = stamp
            save(path, latest)
    except (OSError, ValueError):
        pass  # without a receipt it is only hashed once more


def failed_copies(path):
    """The copies of one catalog file that did not pass, newest first."""
    path = Path(path)
    found = []
    try:
        entries = list(path.parent.iterdir())
    except OSError:
        return []
    for entry in entries:
        rest = entry.name[len(path.name):] if entry.name.startswith(path.name) else ''
        if rest and FAILED.fullmatch(rest) and entry.is_file() and not entry.is_symlink():
            found.append((int(rest.rsplit('-', 1)[1]), entry))
    return [entry for _, entry in sorted(found, key=lambda item: item[0], reverse=True)]


def prune_failed(root, spec, row, keep):
    """Keep at most `keep` failed copies of one of this pack's files (the newest, for diagnosis): one while it is not
    in place, none once it is. Nothing but this catalog's own names in its own folder is touched."""
    if not _ours(spec, row):
        return
    for entry in failed_copies(_target(root, row))[keep:]:
        try:
            entry.unlink()
        except OSError:
            pass


def _missing_rows(root, spec, mode):
    ledger = _ledger(root, spec)
    return [row for row in rows(spec, mode=mode) if not _receipted(root, row, ledger)]


def missing_bytes(root, spec=None, *, mode=DEFAULT_MODE):
    """The full size of the files that have not been verified yet."""
    return sum(row['bytes'] for row in _missing_rows(io_path(root), spec or catalog(), mode))


def default_present(root, spec=None, *, mode=DEFAULT_MODE):
    """Whether the files the engine reads are the ones that passed their sha256, unchanged since (the receipt).

    The licence and notes are fetched with them, and nothing depends on reading them.
    """
    spec = spec or catalog()
    root = io_path(root)
    ledger = _ledger(root, spec)
    return all(_receipted(root, row, ledger) for row in rows(spec, mode=mode) if row.get('engine'))


def default_files(root=None, spec=None, *, mode=DEFAULT_MODE):
    """Map verified engine files for the selected mode to their catalog names."""
    spec = spec or catalog()
    root = io_path(root if root is not None else model_folder())
    ledger = _ledger(root, spec)
    found, absent = {}, []
    for row in rows(spec, mode=mode):
        if not row.get('engine'):
            continue
        path = _target(root, row)
        there = _receipted(root, row, ledger)
        if not there:
            identity = _verified_identity(path, row, lambda: None)
            if identity is not None:
                _receive(root, spec, ledger, row, identity)
                there = True
        if there:
            found[row['file']] = path
        else:
            absent.append(row['file'])
    if absent:
        raise FileNotFoundError(MISSING + 'install the fastest mode files (missing or incomplete: ' + ', '.join(absent) + ')')
    return found


def selected_files(root, mode, *, engine_root=None):
    from .model_components import root_for_model_dir, selected_files as component_files
    from .model_components_generation import PI_SLOTS
    _mode(mode)
    return component_files(engine_root or root_for_model_dir(root), PI_SLOTS[mode])[1]


def present(root, spec=None, *, mode=DEFAULT_MODE):
    return bool(selected_files(root, mode)) or default_present(root, spec, mode=mode)


def files(root=None, spec=None, *, mode=DEFAULT_MODE, engine_root=None):
    root = model_folder() if root is None else root
    selected = selected_files(root, mode, engine_root=engine_root)
    if selected:
        from .fast_tier import CONSTANTS, FOLDER, mode_spec
        try:
            return {f'{FOLDER}/{name}': selected[name] for name in (mode_spec(mode)['adapter'], CONSTANTS)}
        except KeyError as error:
            raise ValueError('The selected pi-H3 component is incomplete.') from error
    return default_files(root, spec, mode=mode)


def _verified_identity(path, row, check, progress=None):
    from .network import hash_file

    def checked(done, total):
        check()
        if progress is not None:
            progress(done, total)

    try:
        if not path.is_file() or path.stat().st_size != row['bytes']:
            return None
        identity = _identity(path, row)
        if hash_file(path, progress=checked) == row['sha256'] and _identity(path, row) == identity:
            return identity
    except InterruptedError:
        raise
    except OSError:
        return None


def _partial(path):
    return path.with_name(path.name + '.partial')


def _resume_bytes(root, todo):
    retained = 0
    for row in todo:
        partial = _partial(_target(root, row))
        try:
            if partial.is_file():
                retained += min(partial.stat().st_size, row['bytes'])
        except OSError:
            pass  # another process may have just committed this file
    return retained


def _engine_lock_path(engine_root):
    from .paths import data_root
    return io_path(Path(engine_root or data_root()) / DOWNLOAD_LOCK)


def _pack_lock_path(job_path):
    return job_path.with_name(DOWNLOAD_LOCK)


@contextmanager
def _download_lease(job_path, engine_root):
    """Hold the physical model-folder lease and the existing lease used by older engines."""
    pack, engine = _pack_lock_path(job_path), _engine_lock_path(engine_root)
    with runtime_lock(pack, inherit=False):
        with nullcontext() if pack == engine else runtime_lock(engine, inherit=False):
            yield


@contextmanager
def _standalone_download_lease(path, engine_root):
    with ExitStack() as lease:
        with _job_lease(path):
            _jobs(path, recover=True)
            lease.enter_context(_download_lease(path, engine_root))
        yield


def install(root, progress=None, check=None, *, engine_root=None, spec=None, mode=DEFAULT_MODE, _leased=False):
    """Fetch what is missing into root; a file that is not the pinned one is kept under another name."""
    from . import network
    from .image_gen.image_pack import PackError as TransferError, _download_plan, _transfer
    from .failure_details import image_pack_error_code
    spec = spec or catalog()
    selected = rows(spec, mode=mode)
    progress = progress or (lambda value: None)
    check = check or (lambda: None)
    root = io_path(root)
    try:
        with nullcontext() if _leased else _standalone_download_lease(_jobs_path(root, spec), engine_root):
            ledger = _ledger(root, spec, retain=True)
            done, total, transferred = 0, 0, 0

            def verifying(current, size):
                progress(dict(phase='verify', done=transferred, total=total, verify_done=current, verify_total=size))

            todo = []
            for row in selected:
                check()
                path = _target(root, row)
                if _receipted(root, row, ledger):
                    prune_failed(root, spec, row, 0)  # the right copy is here: the failed ones go
                    continue
                identity = _verified_identity(path, row, check, verifying)
                if identity is not None:
                    _receive(root, spec, ledger, row, identity)
                    prune_failed(root, spec, row, 0)
                    continue
                if path.exists() or path.is_symlink():
                    # Another feature's file is never moved: a different copy there stops the download.
                    if not _ours(spec, row):
                        raise PackError('integrity_failed')
                    network.retain_partial(path, 'rejected')
                prune_failed(root, spec, row, 1)  # at most the newest failed copy stays while it is not in place
                todo.append(row)
            if not todo:
                return 'present'
            if any(row.get('revision', spec['revision']) is None for row in todo):
                raise PackError('not_published')  # nothing is fetched from anywhere but our copies
            root.mkdir(parents=True, exist_ok=True)
            total = sum(row['bytes'] for row in todo)
            if disk_space.free_bytes(root) < total - _resume_bytes(root, todo) + RESERVE:
                raise PackError('disk_space')
            plan = _download_plan(engine_root)
            plan.update(quiet=True, resource_check=check)

            def downloading(current):
                nonlocal transferred
                transferred = current
                progress(dict(phase='download', done=current, total=total, verify_done=None, verify_total=None))

            downloading(0)
            for row in todo:
                check()
                path = _target(root, row)
                path.parent.mkdir(parents=True, exist_ok=True)

                def moved(current, size, speed, **kwargs):
                    downloading(done + current)
                    check()

                def diagnostic(value):
                    if value.get('phase') == 'verify':
                        verifying(value.get('verify_done') or 0, value.get('verify_total') or row['bytes'])

                try:
                    _transfer(plan, remote(spec, row), path, moved, check, diagnostic=diagnostic)
                finally:
                    if not path.is_file():
                        prune_failed(root, spec, row, 1)  # each failed copy is the whole file: one is enough to look at
                _receive(root, spec, ledger, row)  # network.download checked its sha256 before naming it
                prune_failed(root, spec, row, 0)
                # The record of which source the partial came from is only for resuming it.
                _partial(path).with_name(path.name + '.partial.source.json').unlink(missing_ok=True)
                done += row['bytes']
                downloading(done)
            return 'installed'
    except (PackError, InterruptedError):
        raise
    except TransferError as error:
        raise PackError(error.reason) from error
    except BlockingIOError as error:
        raise PackError('busy') from error
    except Exception as error:
        raise PackError(image_pack_error_code(error)) from error


def _jobs_path(root, spec):
    return _ledger_path(root, spec).with_name(JOB_LEDGER)


def _job_lease(path):
    return runtime_lock(path.with_suffix('.lock'), inherit=False, wait_seconds=5)


def _download_running(path):
    """Probe the existing lease without creating files on a status request."""
    try:
        with _pack_lock_path(path).open('rb') as stream:
            if sys.platform == 'win32':
                windows_shared_lock(stream.fileno())
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except FileNotFoundError:
        return False
    except OSError:
        return True  # held, or unavailable to probe safely
    return False


def _job_records(path):
    ledger = load_ledger(path, retain=False)
    records = ledger.get('records')
    records = {mode: value for mode, value in records.items() if isinstance(value, dict)} if isinstance(records, dict) else {}
    active = ledger.get('active')
    if not isinstance(active, str) or records.get(active, {}).get('state') != 'running':
        active = None
    return dict(active=active, records=records)


def _jobs(path, *, recover=False):
    """Read the job records and treat a running job without its lease as interrupted."""
    state = _job_records(path)
    active = state['active']
    if not active:
        return state
    running = _download_running(path)
    latest = _job_records(path)
    # Re-read because a worker may have committed its terminal state, or the next worker
    # may already own the lease. A new worker always settles the old record before acquiring it.
    if not running and latest['active'] == active and latest['records'][active].get('job_id') == state['records'][active].get('job_id'):
        latest['records'][active].update(state='failed', reason='interrupted', cancelling=False)
        latest['active'] = None
        if recover:
            save(path, latest)
    return latest


class Installer:
    """A shared model-folder job with a worker in either the launcher or ComfyUI."""

    def __init__(self, root=None, engine_root=None, machine=None):
        self.root, self.engine_root, self.machine = root, engine_root, machine
        self.lock = threading.Lock()
        self.mode = self.thread = self.stop = self.job_path = self.job_id = None

    def _location(self):
        bound_engine, machine = _installation() if self.machine is None else (None, self.machine)
        machine = machine() if callable(machine) else machine
        root = self.root() if callable(self.root) else self.root
        return io_path(root if root is not None else _model_folder(machine)), self.engine_root or bound_engine, machine

    def state(self, mode=DEFAULT_MODE):
        _mode(mode)
        root, _, _ = self._location()
        return dict(_jobs(_jobs_path(root, catalog()))['records'].get(mode, IDLE))

    def status(self, mode=DEFAULT_MODE):
        from .model_components import ComponentError
        spec = catalog()
        root, engine, machine = self._location()
        state = _jobs(_jobs_path(root, spec))
        compatibility = setup_compatibility(machine)
        try:
            todo = [] if selected_files(root, mode, engine_root=engine) else _missing_rows(root, spec, mode)
        except ComponentError:
            compatibility = dict(compatible=False, compatibility_reason='selected_component_changed')
            todo = rows(spec, mode=mode)
        return dict(bytes=sum(row['bytes'] for row in todo), resume_bytes=_resume_bytes(root, todo),
                    present=not any(row.get('engine') for row in todo), published=published(spec, mode=mode),
                    blocked_by=state['active'] if state['active'] != mode else None,
                    install=dict(state['records'].get(mode, IDLE)), **compatibility)

    def start(self, mode=DEFAULT_MODE):
        _mode(mode)
        spec = catalog()
        root, engine, machine = self._location()
        path = _jobs_path(root, spec)
        compatibility = setup_compatibility(machine)
        if compatibility['compatible'] is False:
            raise PackError(compatibility['compatibility_reason'])

        try:
            with self.lock, _job_lease(path), ExitStack() as lease:
                state = _jobs(path, recover=True)
                if state['active'] == mode:
                    return dict(state['records'][mode])
                if state['active'] or self.thread and self.thread.is_alive():
                    raise PackError('busy')
                # The transfer lease is nonblocking. We publish the job before releasing the state lease,
                # so a second caller can always identify the owner mode when it acquires the state lease.
                lease.enter_context(_download_lease(path, engine))
                self.mode, self.stop, self.job_path, self.job_id = mode, threading.Event(), path, uuid4().hex
                record = dict(IDLE, state='running', started=time.time(), job_id=self.job_id)
                state['active'] = mode
                state['records'][mode] = record
                save(path, state)
                owner_lease = lease.pop_all()
                self.thread = threading.Thread(
                    target=self._run, name='freevideo-fast-pack', daemon=True,
                    args=(root, engine, spec, path, mode, self.job_id, self.stop, owner_lease))
                try:
                    self.thread.start()
                except BaseException:
                    owner_lease.close()
                    raise
                return dict(record)
        except BlockingIOError as error:
            raise PackError('busy') from error
        except OSError as error:
            from .failure_details import image_pack_error_code
            raise PackError(image_pack_error_code(error)) from error

    def cancel(self, mode=DEFAULT_MODE, job_id=None):
        _mode(mode)
        root, _, _ = self._location()
        return self._cancel(_jobs_path(root, catalog()), mode, job_id)

    @staticmethod
    def _cancel(path, mode, job_id):
        state = _jobs(path)
        if state['active'] != mode:
            return dict(state['records'].get(mode, IDLE))
        with _job_lease(path):
            state = _jobs(path)
            value = state['records'].get(mode, IDLE)
            if state['active'] == mode and (job_id is None or value.get('job_id') == job_id):
                value['cancelling'] = True
                save(path, state)
            return dict(value)

    def owns_download(self):
        with self.lock:
            return bool(self.thread and self.thread.is_alive())

    def cancel_owned(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                return
            self.stop.set()
            path, mode, job_id = self.job_path, self.mode, self.job_id
        try:
            self._cancel(path, mode, job_id)
        except OSError:
            pass  # the local worker already has the stop signal

    def close(self, timeout=5):
        try:
            self.cancel_owned()
        finally:
            if self.thread:
                self.thread.join(timeout)

    @staticmethod
    def _run(root, engine, spec, path, mode, job_id, stop, lease):
        next_check = next_report = 0.
        latest = {}

        def check():
            nonlocal next_check
            now = time.monotonic()
            if now >= next_check:
                next_check = now + REPORT_INTERVAL
                value = _job_records(path)['records'].get(mode, {})
                if value.get('job_id') == job_id and value.get('cancelling'):
                    stop.set()
            if stop.is_set():
                raise InterruptedError('cancelled')

        def report(*, finished=None, reason=None):
            with _job_lease(path):
                state = _job_records(path)
                value = state['records'].get(mode, {})
                if value.get('job_id') != job_id:
                    return
                value.update(latest)
                if finished:
                    value.update(state=finished, reason=reason, cancelling=False)
                    state['active'] = None
                save(path, state)

        def progress(value):
            nonlocal next_report
            now = time.monotonic()
            phase_changed = value.get('phase') != latest.get('phase')
            latest.update(value)
            if phase_changed or now >= next_report:
                next_report = now + REPORT_INTERVAL
                report()

        with lease:
            try:
                install(root, progress, check, engine_root=engine, spec=spec, mode=mode, _leased=True)
            except InterruptedError:
                report(finished='cancelled')
            except PackError as error:
                report(finished='failed', reason=error.reason)
            except Exception:
                report(finished='failed', reason='failed')
            else:
                report(finished='done')


def register():
    """Expose install and cancel for a selected mode, defaulting to fast2."""
    import asyncio
    from aiohttp import web
    from server import PromptServer
    server = PromptServer.instance
    if server is None or getattr(server, '_freevideo_fast_pack', None) is not None:
        return
    installer = server._freevideo_fast_pack = Installer()

    def reply(value, status=200):
        return web.json_response(value, status=status, headers={'Cache-Control': 'no-store'})

    @server.routes.get('/freevideo/fast-pack')
    async def fast_pack_status(request):
        try:
            mode = _mode(request.query.get('mode', DEFAULT_MODE))
        except ValueError:
            return reply({'reason': 'failed'}, 400)
        return reply(await asyncio.to_thread(installer.status, mode))

    @server.routes.post('/freevideo/fast-pack')
    async def fast_pack_install(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {'action', 'mode', 'job_id'} or body.get('action') not in ('install', 'cancel'):
                raise ValueError('Invalid request')
            mode = _mode(body.get('mode', DEFAULT_MODE))
            if 'job_id' in body and (body['action'] != 'cancel' or not isinstance(body['job_id'], str)
                                    or not re.fullmatch(r'[0-9a-f]{32}', body['job_id'])):
                raise ValueError('Invalid download job')
        except (ValueError, TypeError):
            return reply({'reason': 'failed'}, 400)
        if body['action'] == 'cancel':
            arguments = (mode, body['job_id']) if 'job_id' in body else (mode,)
            try:
                await asyncio.to_thread(installer.cancel, *arguments)
            except BlockingIOError:
                return reply({'reason': 'busy'}, 409)
            except OSError:
                return reply({'reason': 'failed'}, 409)
        else:
            state = await asyncio.to_thread(installer.status, mode)
            if state.get('compatible') is False:
                reason = state['compatibility_reason']
                return reply(dict(compatible=False, compatibility_reason=reason, reason=reason), 409)
            if not state['present']:
                try:
                    await asyncio.to_thread(installer.start, mode)
                except PackError as error:
                    return reply({'reason': error.reason}, 409)
        return reply(await asyncio.to_thread(installer.status, mode))
