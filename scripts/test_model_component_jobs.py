"""Exercise model import and selection with small files and no GPU or network."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from freevideo_engine import comfy_setup, model_component_jobs as jobs, model_components as models
from freevideo_engine import model_components_visual as visual
from freevideo_engine.locking import runtime_lock


class ModelComponentCatalogTest(unittest.TestCase):
    def test_catalog_and_status_load_without_inference_dependencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = {'root': str(root), 'model_root': str(root / 'models'), 'device_backend': 'cuda',
                       'base': str(root / 'models' / 'h3-base')}
            rows = {row['id']: row for row in jobs.ComponentJobs(root, machine).status()['slots']}
            self.assertGreaterEqual(len(rows), 47)
            self.assertEqual(rows['qwen-image-2.1-turbo-text-encoder']['manager_component'], 'text_encoder')
            self.assertEqual(rows['prompt-vlm']['group'], 'prompt')
            self.assertEqual(rows['prompt-vlm']['local_selection'], ('file', 'directory'))
            self.assertFalse(rows['prompt-vlm']['local_uses_builtin_files'])
            self.assertEqual(rows['depth-reference']['local_selection'], ('file',))
            fast = rows['pi-h3-2-step']
            self.assertEqual(fast['local_selection'], ('directory',) if fast['supported'] else ())
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

    def test_prompt_file_picker_authorizes_its_complete_bundle_for_one_slot(self):
        prompt = next(slot for slot in visual.slots(self.engine, {}) if slot.id == 'prompt-vlm')
        self.catalog[prompt.id] = prompt
        self.machine['python'] = 'fixture-python'
        setup = SimpleNamespace(source=self.root, runner=None, _local_picker=Lock(),
                                component_jobs=lambda *_: self.manager,
                                component_libraries=lambda _: (self.library,))
        folder = self.root / 'prompt 模型'
        folder.mkdir()
        metadata = visual.PROMPT_REQUIRED | {'tokenizer.json'}
        source = folder / 'Qwen3-VL.safetensors'
        for name in metadata | {source.name}:
            (folder / name).write_bytes(b'fixture')
        def action(slot, **payload):
            return comfy_setup.Setup.component_action(setup, dict(slot=slot.id, **payload), None, None)
        with (patch.object(comfy_setup, 'os', SimpleNamespace(name='nt')),
              patch('freevideo_engine.local_model_picker.select', return_value=str(source)) as pick):
            self.assertEqual(action(prompt, action='pick', kind='file')['path'], str(source))
            pick.assert_called_once_with(self.machine['python'], self.root, 'prompt-model')
        with self.assertRaisesRegex(ValueError, 'configured model library'):
            action(self.slot, action='inspect', source='local', path=str(source))
        candidate = action(prompt, action='inspect', source='local', path=str(source))['candidate']
        self.assertEqual(set(candidate['files']), metadata | {visual.PROMPT_SINGLE})
        with self.assertRaisesRegex(ValueError, 'configured model library'):
            action(prompt, action='inspect', source='local', path=str(source))

    def test_prompt_shard_index_and_file_require_the_whole_bundle(self):
        prompt = next(slot for slot in visual.slots(self.engine, {}) if slot.id == 'prompt-vlm')
        index = self.library / visual.PROMPT_INDEX
        first, second = 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'
        def inspect(path):
            return self.manager.inspect(prompt, {'source': 'local', 'path': str(path)}, (self.library,), '')
        for name in visual.PROMPT_REQUIRED | {'tokenizer.json', visual.PROMPT_INDEX, first}:
            (self.library / name).write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'complete set'):
            inspect(index)
        (self.library / second).write_bytes(b'fixture')
        self.assertEqual(inspect(index)['files'], inspect(self.library)['files'])
        self.assertEqual(inspect(self.library / first)['files'], inspect(self.library)['files'])
        single = self.source('alternative.safetensors')
        self.assertEqual(set(inspect(single)['files']), visual.PROMPT_REQUIRED | {'tokenizer.json', visual.PROMPT_SINGLE})
        (self.library / 'config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing prompt model files: config.json'):
            inspect(index)

    def standalone_prompt(self):
        from freevideo_engine.prompt_vlm import assets

        self.slot = next(slot for slot in visual.slots(self.engine, {}) if slot.id == 'prompt-vlm')
        self.catalog.clear()
        self.catalog[self.slot.id] = self.slot
        folder = self.engine / 'stock'
        folder.mkdir()
        metadata = sorted(visual.PROMPT_REQUIRED | {'tokenizer.json'})
        contents = {name: b'{}' for name in metadata}
        contents[visual.PROMPT_SINGLE] = self.prompt_weights('expected')
        pins = []
        for name, data in contents.items():
            (folder / name).write_bytes(data)
            pins.append(dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        for mock in (patch.object(assets, 'directory', return_value=folder),
                     patch.object(assets, 'catalog', return_value={'files': pins}),
                     patch.object(assets, 'default_ready', return_value=True)):
            mock.start()
            self.addCleanup(mock.stop)
        return folder, metadata

    @staticmethod
    def prompt_weights(name, value=b'\0\0'):
        header = json.dumps({name: dict(dtype='BF16', shape=[len(value) // 2], data_offsets=[0, len(value)])}).encode()
        return len(header).to_bytes(8, 'little') + header + value

    def test_prompt_standalone_copies_selected_weights_with_pinned_builtin_metadata(self):
        stock, metadata = self.standalone_prompt()
        data = self.prompt_weights('expected', b'\0\1')
        self.slot = replace(self.slot, validate=lambda folder: self.assertEqual(
            (folder / visual.PROMPT_SINGLE).read_bytes(), data))
        self.catalog[self.slot.id] = self.slot
        source = self.source('custom.safetensors', data)
        candidate = self.inspect_local(source)
        self.assertTrue(self.status()['slots'][0]['local_uses_builtin_files'])
        self.assertEqual(candidate['builtin_files'], metadata)
        self.assertEqual(set(candidate['files']), set(metadata) | {visual.PROMPT_SINGLE})
        self.assertNotIn('local_path', candidate)
        self.manager.perform(self.slot, 'install', {'candidate_id': candidate['id']}, '')
        identity = self.finish()['id']
        folder, receipt = models.receipt(self.engine, self.slot.id, identity)
        self.assertEqual(self.status()['slots'][0]['active_id'], models.BUILTIN)
        self.assertEqual((folder / visual.PROMPT_SINGLE).read_bytes(), data)
        self.assertEqual(receipt['files']['config.json']['sha256'], hashlib.sha256(b'{}').hexdigest())
        self.assertEqual(receipt['builtin_files'], metadata)
        (source.parent / 'config.json').write_bytes(b'also present now')
        (folder / 'config.json').unlink()
        self.action('repair', identity)
        self.assertEqual((folder / 'config.json').read_bytes(), (stock / 'config.json').read_bytes())

    def test_prompt_standalone_rejects_invalid_weights_and_damaged_or_partial_metadata(self):
        stock, _ = self.standalone_prompt()
        with self.assertRaisesRegex(ValueError, 'valid BF16'):
            self.inspect_local(self.source())
        for data in (self.prompt_weights('different'), self.prompt_weights('expected', b'\0' * 4)):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                self.inspect_local(self.source(content=data))
        valid = self.source(content=self.prompt_weights('expected'))
        with patch('freevideo_engine.prompt_vlm.assets.default_ready', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Install the built-in Qwen3-VL model'):
                self.inspect_local(valid)
        (stock / 'config.json').write_bytes(b'[]')
        with self.assertRaisesRegex(ValueError, 'built-in config.json file is missing or damaged'):
            self.inspect_local(valid)
        (valid.parent / 'config.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'Missing prompt model files'):
            self.inspect_local(valid)

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
