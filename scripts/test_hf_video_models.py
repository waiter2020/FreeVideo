"""CPU checks for Hugging Face video model staging and activation."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from freevideo_engine import hf_video_models as models
from freevideo_engine.local_model_copy import source_record
from freevideo_engine.locking import runtime_lock


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class VideoModelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.machine_path = self.root / 'machine.json'
        self.official_cache = self.root / 'prepared' / 'official'
        self.base = self.root / 'models' / 'h3-base'
        self.checkpoint = self.root / 'models' / 'stage-dmd-step-250'
        self.machine = {
            'ready': True, 'cache': str(self.official_cache), 'base': str(self.base),
            'checkpoint': str(self.checkpoint), 'python': 'python', 'prepared_format': 'fp8',
            'gpu_uuid': 'GPU-fixture', 'compatibility_field': {'value': 1},
        }
        write_json(self.machine_path, self.machine)
        write_json(self.base / 'transformer' / 'config.json', {'num_layers': 0})
        write_json(self.checkpoint / 'model_spec.json', {})
        write_json(self.official_cache / 'manifest.json', {'precision': 'fp8', 'scale_granularity': 'rowwise'})
        self.manager = models.VideoModels(self.root)

    def test_managed_video_pins_shared_encoder_and_keeps_explicit_encoder_arguments(self):
        from freevideo_engine.managed import command_for
        machine = dict(self.machine, root=str(self.root), comfy_python='encoder-python', comfy_root='comfy',
                       model_paths='paths.yaml', encoder='stock.safetensors')
        with patch('freevideo_engine.image_components.shared_encoder', return_value={'id': 'c' * 24}):
            command = command_for(machine, 'generate', [])
            self.assertEqual(command[command.index('--encoder-component') + 1], 'c' * 24)
            explicit = command_for(machine, 'generate', ['--encoder', 'other.safetensors'])
            self.assertNotIn('--encoder-component', explicit)
            rolled_back = command_for(machine, 'generate', ['--encoder-component', 'builtin'])
            self.assertEqual(rolled_back.count('--encoder-component'), 1)
            self.assertEqual(rolled_back[-1], 'builtin')

    def test_worker_resolves_the_explicit_shared_component_instead_of_reloading_active(self):
        from freevideo_engine.encoder_checkpoint import encoder_resident_key, resolve_request_encoder
        from freevideo_engine.long_paths import plain
        encoder = self.root / 'candidate.safetensors'
        encoder.write_bytes(b'candidate')
        record = dict(id='c' * 24, path=str(encoder), sha256=sha256(b'candidate').hexdigest(), label='Candidate')
        with (patch('freevideo_engine.paths.data_root', return_value=self.root),
              patch('freevideo_engine.image_components.shared_encoder', return_value=record) as resolve):
            path, checkpoint = resolve_request_encoder(None, 'stock.safetensors', record['id'])
        self.assertEqual(resolve.call_args.kwargs, {'identity': record['id']})
        self.assertEqual(Path(plain(path)), encoder)
        self.assertEqual(checkpoint['component_sha256'], record['sha256'])
        self.assertEqual(encoder_resident_key(path, self.root), encoder_resident_key(encoder, self.root))

    def _candidate(self, identity='a' * 24):
        candidate = dict(id=identity, repo='example/vdn', label='vdn', revision='f' * 40, kind='prepared',
                         format='fp8_rowwise', cache='cache', files=[], bytes=0)
        self.manager.pending[identity] = candidate
        folder = models.profile_folder(self.root, identity)
        cache = folder / 'cache'
        cache.mkdir(parents=True)
        payload = b'tensor data'
        (cache / 'root.safetensors').write_bytes(payload)
        manifest = dict(precision='fp8', scale_granularity='rowwise', groups=[
            dict(file='root.safetensors', group='root', bytes=len(payload), sha256=sha256(payload).hexdigest())
        ])
        write_json(cache / 'manifest.json', manifest)
        return candidate, cache, manifest

    def _wait_job(self):
        self.manager.thread.join(timeout=5)
        self.assertFalse(self.manager.thread.is_alive())
        return self.manager.status()['task']

    def _start(self, candidate, manifest, smoke):
        with (patch('freevideo_engine.hf_video_catalog.validate_import'),
              patch('freevideo_engine.provision.verify_cache', return_value=manifest),
              patch.object(self.manager, '_child'),
              patch.object(self.manager, '_smoke', side_effect=smoke)):
            self.manager.start(candidate['id'])
            return self._wait_job()

    def test_smoke_precedes_install_and_activation_preserves_default(self):
        candidate, cache, manifest = self._candidate()

        def smoke(machine, folder, staged_cache, descriptor):
            self.assertEqual(staged_cache, cache)
            self.assertEqual(Path(machine['base']), self.base)
            self.assertFalse((folder / models.PROFILE_FILE).exists())
            self.assertEqual(self.manager.status()['active']['id'], models.OFFICIAL_ID)

        task = self._start(candidate, manifest, smoke)
        self.assertEqual(task['phase'], 'ready', task.get('error'))
        self.assertEqual(models._read(self.machine_path), self.machine)
        status = self.manager.activate(candidate['id'])
        current = models._read(self.machine_path)
        self.assertEqual(status['active']['id'], candidate['id'])
        self.assertEqual(Path(current['cache']), cache)
        self.assertEqual(current['base'], self.machine['base'])
        self.assertEqual(current['compatibility_field'], self.machine['compatibility_field'])
        self.assertEqual(len(models.VideoModels(self.root).status()['installed']), 2)
        self.manager.activate(models.OFFICIAL_ID)
        self.assertEqual(models._read(self.machine_path), self.machine)
        self.assertTrue((cache / 'root.safetensors').is_file())

    def test_local_raw_copy_converts_owned_source_before_activation(self):
        source = self.root / 'external weights.safetensors'
        source.write_bytes(b'raw source')
        target = 'h3-base/transformer/diffusion_pytorch_model.safetensors'
        row = source_record(source, target, root=self.root)
        chosen = self.manager._remember_local(dict(source_kind='local', kind='raw', label='Local H3', format='raw',
                                                   files=[row], bytes=row['bytes'], copy_bytes=row['bytes'],
                                                   required_bytes=row['bytes'], base='h3-base', checkpoint='stage-dmd-step-250'))
        manifest = dict(precision='fp8', scale_granularity='rowwise', groups=[])

        def convert(machine, folder, name, args, **_):
            self.assertEqual(name, 'preparation')
            self.assertEqual((folder / target).read_bytes(), b'raw source')
            self.assertEqual(self.manager.status()['active']['id'], models.OFFICIAL_ID)
            write_json(folder / 'converted/profile/manifest.json', manifest)
            write_json(folder / 'conversion.json', {'cache': 'profile'})

        def smoke(machine, folder, cache, descriptor):
            self.assertEqual(cache, folder / 'converted/profile')
            self.assertFalse((folder / models.PROFILE_FILE).exists())

        with (patch('freevideo_engine.local_video_catalog.validate_import') as validate,
              patch('freevideo_engine.provision.verify_cache', return_value=manifest),
              patch.object(self.manager, '_child', side_effect=convert),
              patch.object(self.manager, '_smoke', side_effect=smoke)):
            self.manager.start(chosen['id'], local=True)
            task = self._wait_job()
        self.assertEqual(task['phase'], 'ready', task.get('error'))
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(task['candidate']['format'], 'fp8_rowwise')
        source.unlink()
        self.manager.activate(task['id'])
        current = self.manager._machine()
        self.assertEqual(Path(current['cache']), self.manager._profile_folder(task['id']) / 'converted/profile')
        self.assertEqual(current['base'], self.machine['base'])
        self.manager.activate(models.OFFICIAL_ID)
        self.assertEqual(self.manager._machine(), self.machine)

    def test_smoke_failure_leaves_running_model_and_no_ready_receipt(self):
        candidate, cache, manifest = self._candidate()
        task = self._start(candidate, manifest, RuntimeError('inference failed'))
        self.assertEqual(task['phase'], 'failed')
        self.assertEqual(task['error'], 'inference failed')
        self.assertEqual(models._read(self.machine_path), self.machine)
        self.assertFalse((cache.parent / models.PROFILE_FILE).exists())

    def test_cancel_while_waiting_for_gpu_leaves_candidate_on_disk(self):
        candidate, cache, manifest = self._candidate()
        self.manager.queue_busy = lambda: True
        with (patch('freevideo_engine.hf_video_catalog.validate_import'),
              patch('freevideo_engine.provision.verify_cache', return_value=manifest),
              patch.object(self.manager, '_child'),
              patch.object(self.manager, '_smoke') as smoke):
            self.manager.start(candidate['id'])
            deadline = time.monotonic() + 5
            while self.manager.status()['task']['phase'] != 'waiting_for_gpu':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            self.manager.cancel()
            self.assertEqual(self._wait_job()['phase'], 'cancelled')
        smoke.assert_not_called()
        self.assertTrue((cache / 'root.safetensors').is_file())
        self.assertEqual(models._read(self.machine_path), self.machine)

    def test_activation_rejects_changed_files_and_busy_engine(self):
        candidate, cache, manifest = self._candidate()
        self.assertEqual(self._start(candidate, manifest, lambda *_: None)['phase'], 'ready')
        with runtime_lock(self.root / 'engine.lock', inherit=False):
            with self.assertRaisesRegex(ValueError, 'busy'):
                self.manager.activate(candidate['id'])
        (cache / 'root.safetensors').write_bytes(b'other tensor data')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.manager.activate(candidate['id'])
        self.assertEqual(models._read(self.machine_path), self.machine)

    def test_raw_cache_prepares_with_gpu_lease_using_installed_runtime_paths(self):
        candidate, original_cache, manifest = self._candidate()
        candidate.update(kind='raw', base='raw-base', checkpoint='raw-checkpoint', cache=None)
        folder = original_cache.parent
        converted = folder / 'converted' / 'cached-fp8'
        converted.parent.mkdir()
        original_cache.rename(converted)
        actions = []

        def child(machine, working, name, args, *, descriptor=None, **unused):
            actions.append(name)
            if name == 'preparation':
                self.assertIsNotNone(descriptor)
                self.assertEqual(Path(machine['base']), self.base)
                self.assertEqual(Path(args[-4]), folder / 'raw-base')
                write_json(working / 'conversion.json', {'cache': converted.name})

        with (patch('freevideo_engine.hf_video_catalog.validate_import'),
              patch('freevideo_engine.provision.verify_cache', return_value=manifest),
              patch.object(self.manager, '_child', side_effect=child),
              patch.object(self.manager, '_smoke') as smoke):
            self.manager.start(candidate['id'])
            task = self._wait_job()
            self.assertEqual(task['phase'], 'ready', task.get('error'))
        self.assertEqual(actions, ['download', 'preparation'])
        self.assertEqual(smoke.call_args.args[2], converted)
        self.manager.activate(candidate['id'])
        current = models._read(self.machine_path)
        self.assertEqual(Path(current['cache']), converted)
        self.assertEqual(current['checkpoint'], self.machine['checkpoint'])

    def test_activation_rechecks_queue_after_model_release(self):
        candidate, _, manifest = self._candidate()
        self.assertEqual(self._start(candidate, manifest, lambda *_: None)['phase'], 'ready')
        status = {'busy': False}
        self.manager.queue_busy = lambda: status['busy']
        self.manager.about_to_switch = lambda: status.update(busy=True)
        with self.assertRaisesRegex(ValueError, 'queue'):
            self.manager.activate(candidate['id'])
        self.assertEqual(models._read(self.machine_path), self.machine)

    def test_unsafe_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid video model ID'):
            models.profile_folder(self.root, '../../escape')
        with self.assertRaisesRegex(ValueError, 'Inspect'):
            self.manager.start('b' * 24)


if __name__ == '__main__':
    unittest.main()
