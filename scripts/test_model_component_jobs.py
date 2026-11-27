"""Exercise model import and selection with small files and no GPU or network."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from freevideo_engine import model_component_jobs as jobs, model_components as models
from freevideo_engine.locking import runtime_lock


class ModelComponentCatalogTest(unittest.TestCase):
    def test_catalog_and_status_load_without_inference_dependencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = {'root': str(root), 'model_root': str(root / 'models'), 'device_backend': 'cuda'}
            rows = {row['id']: row for row in jobs.ComponentJobs(root, machine).status()['slots']}
            self.assertGreaterEqual(len(rows), 47)
            self.assertEqual(rows['qwen-image-2.1-turbo-text-encoder']['manager_component'], 'text_encoder')
            self.assertEqual(rows['prompt-vlm']['group'], 'prompt')
            self.assertIn('Speaker identification', rows['campplus']['consumers'])
            self.assertTrue(rows['editor-silero-vad']['default']['bytes'] > 0)


class ModelComponentJobsTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.engine = self.root / 'engine'
        self.engine.mkdir()
        self.library = self.root / 'local 权重'
        self.library.mkdir()
        self.filename = 'model.safetensors'
        self.default = self.engine / self.filename
        self.default.write_bytes(b'checkpoint: builtin')
        self.slot = models.Slot('test-model', 'visual', 'Test model', '测试模型', ('Fixture',),
                                {self.filename: self.default}, 'Safetensors', validate=self.validate)
        self.catalog = {self.slot.id: self.slot}
        self.machine = {'comfy_python': 'fixture-python'}
        self.manager = jobs.ComponentJobs(self.engine, self.machine)
        self.smoke = Mock(side_effect=self.mock_worker)
        for mock in (patch.object(models, 'catalog', return_value=self.catalog),
                     patch.object(self.manager, '_child', self.smoke),
                     patch.dict(os.environ, FREEVIDEO_LOCK_PATH=str(self.engine / 'engine.lock'))):
            mock.start()
            self.addCleanup(mock.stop)
        self.addCleanup(self.stop_worker)

    def validate(self, folder):
        if not (folder / self.filename).read_bytes().startswith(b'checkpoint: '):
            raise ValueError('Invalid fixture checkpoint.')

    def mock_worker(self, machine, workspace, name, args, **kwargs):
        self.assertEqual(machine, self.machine)
        self.assertEqual(name, 'test')
        module, root, slot, folder, result = args[1:]
        self.assertEqual(module, 'freevideo_engine.model_component_jobs')
        self.assertEqual(Path(root), self.engine)
        self.assertEqual(Path(result).parent, workspace)
        self.assertEqual(kwargs['python'], self.machine['comfy_python'])
        self.catalog[slot].validate(Path(folder))

    def stop_worker(self):
        thread = self.manager.thread
        if thread is not None and thread.is_alive():
            self.manager.cancel()
            thread.join(5)

    def status(self):
        return self.manager.status(local=True, native_picker=False)

    def finish(self, phase='ready'):
        self.manager.thread.join(5)
        self.assertFalse(self.manager.thread.is_alive())
        job = self.status()['job']
        self.assertEqual(job['phase'], phase, job.get('error'))
        self.assertFalse(list(models.store(self.engine).glob('import-*')))
        return job

    def source(self, filename='其他 模型.safetensors', content=b'checkpoint: custom'):
        path = self.library / filename
        path.write_bytes(content)
        return path

    def inspect_local(self, source):
        return self.manager.inspect(self.slot, {'source': 'local', 'path': str(source)}, (self.library,), '')

    def install_local(self, source, phase='ready'):
        candidate = self.inspect_local(source)
        self.manager.perform(self.slot, 'install', {'candidate_id': candidate['id']}, '')
        return self.finish(phase)

    def action(self, action, identity=None, *, phase='ready'):
        payload = {'version_id': identity} if identity is not None else {}
        self.manager.perform(self.slot, action, payload, '')
        return self.finish(phase)

    def test_install_copies_local_files_and_requires_explicit_activation(self):
        source = self.source()
        candidate = self.inspect_local(source)
        self.assertEqual(candidate['label'], source.name)
        self.assertEqual(candidate['files'], [self.filename])
        self.assertNotIn('local_path', candidate)
        self.manager.perform(self.slot, 'install', {'candidate_id': candidate['id']}, '')
        identity = self.finish()['id']
        self.assertNotEqual(candidate['id'], identity)
        self.assertEqual(self.status()['slots'][0]['active_id'], models.BUILTIN)
        self.assertEqual(models.selected_file(self.engine, self.slot.id, self.filename, self.default), self.default)
        source_content = source.read_bytes()
        source.rename(source.with_suffix('.moved'))
        self.action('activate', identity)
        self.action('verify', identity)
        reopened = jobs.ComponentJobs(self.engine, self.machine)
        self.assertEqual(reopened.status()['slots'][0]['active_id'], identity)
        owned = models.selected_file(self.engine, self.slot.id, self.filename, self.default)
        self.assertTrue(owned.is_relative_to(self.engine))
        self.assertEqual(owned.read_bytes(), source_content)
        self.assertEqual(models.selected_hash(self.engine, self.slot.id, self.filename), hashlib.sha256(source_content).hexdigest())
        self.assertNotIn('local_path', self.status()['slots'][0]['versions'][0])

    def test_default_delegates_can_acquire_their_existing_setup_lock(self):
        def install(progress, check):
            with runtime_lock(self.engine / 'setup.lock', inherit=False):
                check()
                progress(5, 5)

        self.slot = replace(self.slot, install_default=install)
        self.catalog[self.slot.id] = self.slot
        self.assertEqual(self.action('install', models.BUILTIN)['done_bytes'], 5)

    def test_remove_protects_a_version_referenced_by_a_working_checkpoint(self):
        identity = self.install_local(self.source())['id']
        output = self.root / 'output'
        work = output / 'FreeVideo' / 'day' / 'run'
        artifacts = work / 'video.artifacts'
        artifacts.mkdir(parents=True)
        (artifacts / 'first-pass.pt').touch()
        (work / 'video.request.json').write_text(json.dumps(dict(success=True, preview=True,
            model_components={self.slot.id: identity})), encoding='utf-8')
        self.manager.retained = lambda version: models.retained_use(output, version)
        self.assertIn('saved working checkpoint', self.action('remove', identity, phase='failed')['error'])
        (artifacts / 'first-pass.pt').unlink()
        self.action('remove', identity)

    def test_rejects_unapproved_or_changed_local_files(self):
        outside = self.root / 'unapproved.safetensors'
        outside.write_bytes(b'checkpoint: unapproved')
        with self.assertRaisesRegex(ValueError, 'configured model library'):
            self.inspect_local(outside)
        source = self.source()
        candidate = self.inspect_local(source)
        source.write_bytes(b'checkpoint: edited after inspection')
        self.manager.perform(self.slot, 'install', {'candidate_id': candidate['id']}, '')
        self.assertIn('changed', self.finish('failed')['error'])
        self.assertEqual(self.status()['slots'][0]['versions'], [])
        self.smoke.assert_not_called()

    def test_failed_validation_or_activation_keeps_current_selection(self):
        identity = self.install_local(self.source())['id']
        self.action('activate', identity)
        invalid = self.source('invalid.safetensors', b'not a checkpoint')
        self.assertIn('Invalid fixture', self.install_local(invalid, 'failed')['error'])
        self.assertEqual([version['id'] for version in self.status()['slots'][0]['versions']], [identity])
        other = self.install_local(self.source('other.safetensors', b'checkpoint: second'))['id']
        folder, _ = models.receipt(self.engine, self.slot.id, other)
        (folder / self.filename).write_bytes(b'checkpoint: tampered after installation')
        self.assertIn('changed', self.action('activate', other, phase='failed')['error'])
        self.assertEqual(models.selected_identity(self.engine, self.slot.id), identity)
        damaged = next(version for version in self.status()['slots'][0]['versions'] if version['id'] == other)
        self.assertTrue(damaged['damaged'])
        self.assertEqual(damaged['actions'], ['repair'])

    def test_cancelled_or_failed_reset_keeps_current_selection(self):
        identity = self.install_local(self.source())['id']
        self.action('activate', identity)
        self.manager.queue_busy = lambda: True
        self.manager.perform(self.slot, 'reset', {}, '')
        self.manager.perform(self.slot, 'cancel', {}, '')
        self.finish('cancelled')
        self.assertEqual(models.selected_identity(self.engine, self.slot.id), identity)
        self.manager.queue_busy = lambda: False
        with patch.object(models, 'save', side_effect=OSError('selection is read-only')):
            self.assertIn('read-only', self.action('reset', phase='failed')['error'])
        self.assertEqual(models.selected_identity(self.engine, self.slot.id), identity)
        self.action('reset')
        self.assertEqual(self.status()['slots'][0]['active_id'], models.BUILTIN)

    def test_remove_preserves_active_version_source_and_untracked_files(self):
        source = self.source()
        identity = self.install_local(source)['id']
        folder, _ = models.receipt(self.engine, self.slot.id, identity)
        self.action('activate', identity)
        self.assertIn('active model', self.action('remove', identity, phase='failed')['error'])
        self.assertTrue((folder / self.filename).is_file())
        self.action('reset')
        extra = folder / 'personal-notes.txt'
        extra.write_bytes(b'keep these notes')
        self.assertIn('additional files', self.action('remove', identity, phase='failed')['error'])
        self.assertEqual(extra.read_bytes(), b'keep these notes')
        extra.unlink()
        self.action('remove', identity)
        self.assertFalse(folder.exists())
        self.assertEqual(source.read_bytes(), b'checkpoint: custom')
        self.assertEqual(self.status()['slots'][0]['versions'], [])

    def test_hf_import_pins_commit_and_publishes_only_verified_model_files(self):
        repo = 'example/fixture'
        commit = 'd' * 40
        token = 'hf_TestFixture'
        content = b'checkpoint: Hugging Face'
        checksum = hashlib.sha256(content).hexdigest()
        path = 'subfolder/alternate.safetensors'
        metadata = dict(id=repo, sha=commit, cardData={'license': 'apache-2.0'},
                        siblings=[dict(rfilename=path, size=len(content), lfs={'sha256': checksum, 'size': len(content)})])
        with patch.object(jobs.hub, '_request', return_value=json.dumps(metadata).encode()) as fetch:
            candidate = self.manager.inspect(self.slot, {'source': 'hf', 'repo': repo, 'revision': 'release', 'path': path}, (), token)
        fetch.assert_called_once_with(f'/api/models/{repo}/revision/release?blobs=true', token=token)
        self.assertEqual(candidate['revision'], commit)
        self.assertEqual(candidate['license'], 'apache-2.0')
        self.assertEqual(candidate['files'], [self.filename])

        def worker(machine, folder, name, args, **kwargs):
            if name == 'test':
                return self.mock_worker(machine, folder, name, args, **kwargs)
            self.assertEqual(name, 'download')
            self.assertEqual(machine, self.machine)
            self.assertEqual(kwargs['token'], token)
            self.assertEqual(args[:-1], ['-m', 'freevideo_engine.hf_video_models', 'transfer'])
            pinned = json.loads(Path(args[-1]).read_text(encoding='utf-8'))
            self.assertEqual(pinned['files'], [dict(repo=repo, revision=commit, file=path, target=self.filename,
                                                  bytes=len(content), official_only=True, sha256=checksum)])
            (folder / self.filename).write_bytes(content)
            Path(kwargs['progress']).write_text(json.dumps({'done': len(content)}), encoding='utf-8')

        self.smoke.side_effect = worker
        self.manager.perform(self.slot, 'install', {'candidate_id': candidate['id']}, token)
        identity = self.finish()['id']
        self.assertEqual([call.args[2] for call in self.smoke.call_args_list], ['download', 'test'])
        folder, record = models.receipt(self.engine, self.slot.id, identity)
        self.assertEqual({path.name for path in folder.iterdir()}, {self.filename, models.RECEIPT})
        self.assertEqual((folder / self.filename).read_bytes(), content)
        self.assertEqual(record['revision'], commit)
        self.assertEqual(record['files'][self.filename]['sha256'], checksum)
        self.assertNotIn(token, json.dumps(record))

    def test_default_actions_follow_installation_and_call_slot_hooks(self):
        self.default.unlink()

        def install(progress, check):
            check()
            self.default.write_bytes(b'checkpoint: builtin')
            progress(19, 19)

        downloader = Mock(side_effect=install)
        verifier = Mock()
        self.slot = replace(self.slot, install_default=downloader, verify_default=verifier, default_bytes=19)
        self.catalog[self.slot.id] = self.slot
        self.assertEqual(self.status()['slots'][0]['default']['actions'], ['install'])
        self.action('install', models.BUILTIN)
        self.assertCountEqual(self.status()['slots'][0]['default']['actions'], ['verify', 'repair'])
        self.action('verify', models.BUILTIN)
        self.action('repair', models.BUILTIN)
        self.assertEqual(downloader.call_count, 2)
        verifier.assert_called_once_with()
        self.assertIsNone(models.selected_identity(self.engine, self.slot.id))


if __name__ == '__main__':
    unittest.main()
