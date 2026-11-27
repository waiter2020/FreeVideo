from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from freevideo_engine import comfy_bridge, fast_pack, fast_tier
from freevideo_engine.locking import runtime_lock
from freevideo_engine.long_paths import io_path


def _job_spec():
    payloads = {'fast2': b'two step adapter', 'fast4': b'four step adapter'}
    rows = [dict(file=f'pi-h3/{mode}', bytes=len(data), sha256=sha256(data).hexdigest(), modes=[mode], engine=True)
            for mode, data in payloads.items()]
    return dict(folder='pi-h3', repo='local/test', revision='abc', modelscope=None, files=rows), payloads


def _run_job_fixture():
    root, engine, mode, action = sys.argv[1:]
    spec, payloads = _job_spec()

    def transfer(_plan, row, target, progress, check, **_kwargs):
        data = payloads[mode]
        partial = fast_pack._partial(target)
        if action == 'resume':
            retained = partial.read_bytes()
            assert retained == data[:5]
            with partial.open('ab') as stream:
                stream.write(data[len(retained):])
            partial.replace(target)
            progress(len(data), len(data), 0)
            return
        partial.write_bytes(data[:5])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            progress(5, row['bytes'], 0)
            check()
            time.sleep(.025)
        raise AssertionError('Timed out waiting for cancellation')

    with (patch.object(fast_pack, 'catalog', return_value=spec),
          patch('freevideo_engine.image_gen.image_pack._download_plan', return_value={}),
          patch('freevideo_engine.image_gen.image_pack._transfer', side_effect=transfer),
          patch.object(fast_pack.disk_space, 'free_bytes', return_value=2 * fast_pack.RESERVE)):
        installer = fast_pack.Installer(root=Path(root), engine_root=Path(engine), machine=lambda: None)
        installer.start(mode)
        installer.thread.join(17)
        expected = 'done' if action == 'resume' else 'cancelled'
        assert installer.state(mode)['state'] == expected, installer.state(mode)


@contextmanager
def _start_job_fixture(root, engine, mode, action):
    # A Windows venv's python.exe is a redirector. The crash test must terminate the
    # actual interpreter that holds the leases, which only needs the standard library.
    executable = getattr(sys, '_base_executable', sys.executable) if os.name == 'nt' else sys.executable
    process = subprocess.Popen([executable, '-B', '-c',
                                'from scripts.test_fast_pack import _run_job_fixture; _run_job_fixture()',
                                str(root), str(engine), mode, action], cwd=Path(__file__).resolve().parents[1])
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(5)


class FastPackTest(unittest.TestCase):
    def test_compatibility_without_a_capability_grid(self):
        with patch.object(fast_tier.sys, 'platform', 'darwin'):
            self.assertEqual(fast_pack.setup_compatibility(None), dict(compatible=False, compatibility_reason='mac'))
        with patch.object(fast_tier.sys, 'platform', 'win32'):
            for machine, reason in ((dict(prepared_format='fp8'), 'engine'),
                                    (dict(video_model=False), 'engine'),
                                    (dict(gpu_vendor='amd'), 'gpu'),
                                    (dict(hardware=dict(capability=[7, 5])), 'gpu'),
                                    (dict(prepared_format='int8_convrot'), None)):
                with self.subTest(machine=machine):
                    self.assertEqual(fast_pack.setup_compatibility(machine), dict(compatible=reason is None, compatibility_reason=reason))
            self.assertEqual(fast_pack.setup_compatibility(None), dict(compatible=None, compatibility_reason=None))

    def test_server_pack_follows_the_bound_engine_and_download_settings(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop('FREEVIDEO_MODEL_ROOT', None)
            engine_root = Path(directory) / 'bound-engine'
            model_root = Path(directory) / 'bound-models'
            override = Path(directory) / 'override-models'
            fallback = Path(directory) / 'fallback-models'
            with patch.object(comfy_bridge, 'installation', return_value=(engine_root, {'model_root': str(model_root)})):
                self.assertEqual(fast_pack.model_folder(), model_root)
                installer = fast_pack.Installer()
                with patch.object(fast_pack, 'install') as install:
                    os.environ['FREEVIDEO_MODEL_ROOT'] = str(override)
                    self.assertEqual(fast_pack.model_folder(), override)
                    installer.start('fast4')
                    installer.thread.join(5)
                    self.assertEqual(installer.state('fast4')['state'], 'done')
                    self.assertEqual(install.call_args.args[0], io_path(override))
                    self.assertEqual(install.call_args.kwargs,
                                     dict(engine_root=engine_root, mode='fast4', spec=fast_pack.catalog(), _leased=True))
            os.environ.pop('FREEVIDEO_MODEL_ROOT', None)
            with (patch.object(comfy_bridge, 'installation', side_effect=ValueError('setup missing')),
                  patch('freevideo_engine.paths.installed_model_root', return_value=fallback)):
                self.assertEqual(fast_pack.model_folder(), fallback)

    def test_catalog_selects_only_the_requested_adapter(self):
        spec = fast_pack.catalog()
        fast2 = fast_pack.rows(spec)
        fast4 = fast_pack.rows(spec, mode='fast4')
        self.assertEqual(fast2[:-1], fast4[:-1])
        self.assertIn('2step', fast2[-1]['file'])
        self.assertIn('4step', fast4[-1]['file'])
        for mode, row in (('fast2', fast2[-1]), ('fast4', fast4[-1])):
            tier = fast_tier.mode_spec(mode)
            self.assertEqual(Path(row['file']).name, tier['adapter'])
            self.assertEqual((row['bytes'], row['sha256']), (tier['adapter_bytes'], tier['adapter_sha256']))
        sources = {row['id']: row['url'] for row in fast_pack.remote(spec, fast4[-1])['sources']}
        self.assertTrue(sources['hf-mirror'].startswith('https://hf-mirror.com/Lakonik/pi-H3/resolve/'))
        self.assertTrue(sources['official'].endswith(fast4[-1]['file']))
        self.assertIn('modelscope', sources)

    def test_install_reuses_the_shared_receipt(self):
        payloads = {'pi-h3/constants': b'const', 'pi-h3/two': b'two', 'pi-h3/four': b'four'}
        rows = [dict(file=name, bytes=len(data), sha256=sha256(data).hexdigest(), engine=True)
                for name, data in payloads.items()]
        rows[1]['modes'] = ['fast2']
        rows[2]['modes'] = ['fast4']
        spec = dict(folder='pi-h3', repo='local/test', revision='abc', modelscope=None, files=rows)
        transfers = []
        phases = []

        def transfer(_plan, row, target, progress, _check, *, diagnostic):
            transfers.append(row['file'])
            target.write_bytes(payloads[row['file']])
            progress(row['bytes'], row['bytes'], 0)
            diagnostic(dict(phase='verify', verify_done=row['bytes'], verify_total=row['bytes']))

        with TemporaryDirectory() as directory:
            root = Path(directory) / 'models'
            (root / 'pi-h3').mkdir(parents=True)
            local = root / 'pi-h3' / 'constants'
            local.write_bytes(payloads['pi-h3/constants'])
            check = Mock(side_effect=[None, InterruptedError('cancelled')])
            with self.assertRaises(InterruptedError):
                fast_pack.install(root, check=check, spec=spec, engine_root=directory)
            self.assertEqual(local.read_bytes(), payloads['pi-h3/constants'])
            with (patch('freevideo_engine.image_gen.image_pack._download_plan', return_value={}),
                  patch('freevideo_engine.image_gen.image_pack._transfer', side_effect=transfer),
                  patch.object(fast_pack.disk_space, 'free_bytes', return_value=2 * fast_pack.RESERVE)):
                fast_pack.install(root, phases.append, spec=spec, engine_root=directory)
                self.assertTrue(fast_pack.present(root, spec))
                self.assertFalse(fast_pack.present(root, spec, mode='fast4'))
                self.assertEqual(fast_pack.missing_bytes(root, spec, mode='fast4'), 4)
                with self.assertRaisesRegex(FileNotFoundError, 'pi-h3/four'):
                    fast_pack.files(root, spec, mode='fast4')
                fast_pack.install(root, spec=spec, mode='fast4', engine_root=directory)

            self.assertEqual(transfers, ['pi-h3/two', 'pi-h3/four'])
            self.assertIn(dict(phase='verify', verify_done=5, verify_total=5, done=0, total=0), phases)
            self.assertIn(dict(phase='verify', verify_done=3, verify_total=3, done=3, total=3), phases)
            fast_pack._receive(root, spec, {}, rows[2])  # a stale caller cannot erase other receipts
            self.assertTrue(fast_pack.present(root, spec))
            self.assertEqual(set(fast_pack.files(root, spec)), {'pi-h3/constants', 'pi-h3/two'})
            self.assertEqual(set(fast_pack.files(root, spec, mode='fast4')), {'pi-h3/constants', 'pi-h3/four'})
            (root / 'pi-h3' / 'four').write_bytes(b'fake')
            self.assertTrue(fast_pack.present(root, spec))
            self.assertFalse(fast_pack.present(root, spec, mode='fast4'))
            with self.assertRaisesRegex(FileNotFoundError, 'pi-h3/four'):
                fast_pack.files(root, spec, mode='fast4')

    def test_installer_progress_and_cancellation_are_mode_specific(self):
        entered, release = Event(), Event()

        def install(_root, progress, check, *, mode, **_kwargs):
            self.assertEqual(mode, 'fast4')
            progress(dict(done=8, total=14))
            entered.set()
            release.wait(5)
            check()

        with TemporaryDirectory() as directory, patch.object(fast_pack, 'install', side_effect=install):
            installer = fast_pack.Installer(root=lambda: Path(directory) / 'models', engine_root=Path(directory), machine=lambda: None)
            installer.start('fast4')
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(installer.state('fast4')['done'], 8)
                self.assertEqual(installer.state()['state'], 'idle')
                self.assertEqual(installer.cancel()['state'], 'idle')
                self.assertFalse(installer.stop.is_set())
                with self.assertRaisesRegex(fast_pack.PackError, 'busy'):
                    installer.start()
                self.assertTrue(installer.owns_download())
                installer.cancel_owned()
                self.assertTrue(installer.state('fast4')['cancelling'])
            finally:
                release.set()
                installer.thread.join(5)
            self.assertEqual(installer.state('fast4')['state'], 'cancelled')
            self.assertEqual(installer.state()['state'], 'idle')
            self.assertFalse(installer.owns_download())

    def test_job_status_is_read_only_and_uses_the_selected_machine(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'untouched'
            installer = fast_pack.Installer(root=lambda: root, engine_root=Path(directory), machine={'prepared_format': 'fp8'})
            with patch.object(fast_pack, '_installation', side_effect=AssertionError('unexpected host lookup')):
                state = installer.status('fast4')
                self.assertFalse(state['compatible'])
                self.assertEqual(state['compatibility_reason'], 'engine')
                self.assertEqual(state['resume_bytes'], 0)
                self.assertIsNone(state['blocked_by'])
                self.assertEqual(installer.cancel('fast4'), fast_pack.IDLE)
                with self.assertRaisesRegex(fast_pack.PackError, 'engine'):
                    installer.start('fast4')
            self.assertFalse(root.exists())

    def test_completed_job_between_record_and_lease_probe_does_not_appear_interrupted(self):
        running = dict(state='running', job_id='same')
        done = dict(state='done', job_id='same')
        with (patch.object(fast_pack, '_job_records', side_effect=[dict(active='fast4', records={'fast4': running}),
                                                                  dict(active=None, records={'fast4': done})]),
              patch.object(fast_pack, '_download_running', return_value=False)):
            self.assertEqual(fast_pack._jobs(Path('job.json')), dict(active=None, records={'fast4': done}))

    def test_successor_taking_the_lease_does_not_revive_the_previous_job(self):
        running = dict(state='running', job_id='old')
        interrupted = dict(state='failed', reason='interrupted', job_id='old')
        records = [dict(active='fast4', records={'fast4': running}), dict(active=None, records={'fast4': interrupted})]
        with (patch.object(fast_pack, '_job_records', side_effect=records),
              patch.object(fast_pack, '_download_running', return_value=True)):
            self.assertEqual(fast_pack._jobs(Path('job.json')), records[-1])

    def test_previous_engine_lease_still_excludes_downloads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'models'
            engine = Path(directory) / 'engine'
            installer = fast_pack.Installer(root=root, engine_root=engine, machine=lambda: None)
            with runtime_lock(engine / fast_pack.DOWNLOAD_LOCK, inherit=False):
                with self.assertRaisesRegex(fast_pack.PackError, 'busy'):
                    installer.start('fast4')
            self.assertEqual(installer.state('fast4'), fast_pack.IDLE)

    def _wait_for_fixture_progress(self, installer, mode, process):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = installer.status(mode)
            if status['install']['state'] == 'running' and status['install']['done'] == 5:
                return status
            self.assertIsNone(process.poll(), status)
            time.sleep(.025)
        self.fail(f'No cross-process progress for {mode}: {status}')

    def test_cross_process_progress_attach_and_cancel_preserve_resume(self):
        spec, _ = _job_spec()
        with TemporaryDirectory() as directory, patch.object(fast_pack, 'catalog', return_value=spec):
            root = Path(directory) / 'models'
            engine, other_engine = Path(directory) / 'engine', Path(directory) / 'other-engine'
            viewer = fast_pack.Installer(root=root, engine_root=other_engine, machine=lambda: None)
            with _start_job_fixture(root, engine, 'fast4', 'hold') as owner:
                state = self._wait_for_fixture_progress(viewer, 'fast4', owner)
                self.assertEqual(state['resume_bytes'], 5)
                self.assertFalse(state['present'])
                self.assertEqual(state['install']['phase'], 'download')
                self.assertEqual(viewer.status()['blocked_by'], 'fast4')
                self.assertEqual(viewer.start('fast4')['job_id'], state['install']['job_id'])
                self.assertFalse(viewer.owns_download())
                with self.assertRaisesRegex(fast_pack.PackError, 'busy'):
                    viewer.start()
                viewer.close()
                self.assertFalse(viewer.cancel('fast4', '0' * 32)['cancelling'])
                self.assertTrue(viewer.cancel('fast4', state['install']['job_id'])['cancelling'])
                self.assertEqual(owner.wait(8), 0)
            self.assertEqual(viewer.status('fast4')['install']['state'], 'cancelled')
            self.assertEqual(viewer.status('fast4')['resume_bytes'], 5)
            self.assertIsNone(viewer.status()['blocked_by'])
            with _start_job_fixture(root, other_engine, 'fast4', 'resume') as worker:
                self.assertEqual(worker.wait(8), 0)
            self.assertTrue(viewer.status('fast4')['present'])
            self.assertFalse(viewer.status()['present'])
            self.assertNotEqual(viewer.status('fast4')['install']['job_id'], state['install']['job_id'])

    def test_killed_owner_is_interrupted_and_old_cancel_cannot_stop_its_successor(self):
        spec, _ = _job_spec()
        with TemporaryDirectory() as directory, patch.object(fast_pack, 'catalog', return_value=spec):
            root, engine = Path(directory) / 'models', Path(directory) / 'engine'
            viewer = fast_pack.Installer(root=root, engine_root=engine, machine=lambda: None)
            with _start_job_fixture(root, engine, 'fast2', 'hold') as owner:
                first = self._wait_for_fixture_progress(viewer, 'fast2', owner)
                owner.kill()
                owner.wait(5)
            stopped = viewer.status()
            self.assertFalse(stopped['present'])
            self.assertEqual(stopped['install']['reason'], 'interrupted')
            self.assertEqual(stopped['resume_bytes'], 5)
            self.assertIsNone(viewer.status('fast4')['blocked_by'])
            with _start_job_fixture(root, engine, 'fast2', 'hold') as successor:
                second = self._wait_for_fixture_progress(viewer, 'fast2', successor)
                self.assertNotEqual(first['install']['job_id'], second['install']['job_id'])
                self.assertFalse(viewer.cancel('fast2', first['install']['job_id'])['cancelling'])
                self.assertTrue(viewer.cancel()['cancelling'])
                self.assertEqual(successor.wait(8), 0)

    def test_four_step_resume_keeps_the_partial_and_reserves_only_the_remaining_bytes(self):
        content = b'pi-h3-four-step'
        row = dict(file='pi-h3/four', bytes=len(content), sha256=sha256(content).hexdigest(), modes=['fast4'], engine=True)
        spec = dict(folder='pi-h3', repo='local/test', revision='abc', modelscope=None, files=[row])
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'models'
            path = root / row['file']
            path.parent.mkdir(parents=True)
            partial = path.with_suffix('.partial')
            partial.write_bytes(content[:7])

            def transfer(_plan, entry, target, _progress, _check, **_options):
                self.assertEqual(entry['sha256'], row['sha256'])
                self.assertEqual(target, io_path(path))
                self.assertEqual(partial.read_bytes(), content[:7])
                with partial.open('ab') as download:
                    download.write(content[7:])
                partial.replace(target)

            with (patch('freevideo_engine.image_gen.image_pack._download_plan', return_value={}),
                  patch('freevideo_engine.image_gen.image_pack._transfer', side_effect=transfer) as resumed,
                  patch.object(fast_pack.disk_space, 'free_bytes', return_value=fast_pack.RESERVE + len(content) - 7)):
                fast_pack.install(root, spec=spec, mode='fast4', engine_root=directory)
                resumed.assert_called_once()
            self.assertTrue(fast_pack.present(root, spec, mode='fast4'))
            self.assertEqual(path.read_bytes(), content)


class FastPackRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_route_defaults_to_two_steps(self):
        routes = {}

        def register(method):
            def route(path):
                def add(handler):
                    routes[method, path] = handler
                    return handler
                return add
            return route

        server = SimpleNamespace(routes=SimpleNamespace(get=register('GET'), post=register('POST')))
        installer = Mock()
        installer.status.return_value = dict(bytes=1, present=False, published=True, install=fast_pack.IDLE)
        modules = {
            'server': SimpleNamespace(PromptServer=SimpleNamespace(instance=server)),
            'aiohttp': SimpleNamespace(web=SimpleNamespace(json_response=lambda value, status, headers: (status, value))),
        }
        with patch.dict(sys.modules, modules), patch.object(fast_pack, 'Installer', return_value=installer):
            fast_pack.register()
        status = routes['GET', '/freevideo/fast-pack']
        action = routes['POST', '/freevideo/fast-pack']

        async def legacy_request():
            return {'action': 'install'}

        async def four_step_request():
            return {'action': 'cancel', 'mode': 'fast4'}

        async def scoped_four_step_request():
            return {'action': 'cancel', 'mode': 'fast4', 'job_id': 'a' * 32}

        async def invalid_job_request():
            return {'action': 'install', 'job_id': 'a' * 32}

        self.assertEqual(await status(SimpleNamespace(query={})), (200, installer.status.return_value))
        installer.status.assert_called_with('fast2')
        await action(SimpleNamespace(json=legacy_request))
        installer.start.assert_called_once_with('fast2')
        await action(SimpleNamespace(json=four_step_request))
        installer.cancel.assert_called_once_with('fast4')
        await action(SimpleNamespace(json=scoped_four_step_request))
        installer.cancel.assert_called_with('fast4', 'a' * 32)
        self.assertEqual(await action(SimpleNamespace(json=invalid_job_request)), (400, {'reason': 'failed'}))
        self.assertEqual(await status(SimpleNamespace(query={'mode': 'unknown'})), (400, {'reason': 'failed'}))
        installer.start.side_effect = fast_pack.PackError('busy')
        self.assertEqual(await action(SimpleNamespace(json=legacy_request)), (409, {'reason': 'busy'}))
        installer.start.reset_mock()
        installer.status.return_value.update(compatible=False, compatibility_reason='mac')
        self.assertEqual(await action(SimpleNamespace(json=legacy_request)),
                         (409, dict(compatible=False, compatibility_reason='mac', reason='mac')))
        installer.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
