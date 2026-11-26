"""Exercise the local copy lifecycle without GPU inference or network traffic."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from freevideo_engine import hf_image_models as images, local_model_copy
from freevideo_engine.monitoring import save


class LocalModelJobsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / 'user models 中文'
        self.inputs.mkdir()
        self.engine = self.root / 'engine'
        self.engine.mkdir()
        self.family = images.image_model.QWEN
        self.target = 'image_gen/denoiser.safetensors'
        save(self.engine / 'machine.json', dict(ready=True, python='python', comfy_python='comfy-python',
                                                model_root=str(self.root / 'stock'), gpu_uuid='test-gpu'))
        self.manager = images.ImageModels(self.engine)
        self.addCleanup(self._stop)
        for target, options in (
            ('freevideo_engine.hf_image_models._ready', {'return_value': True}),
            ('freevideo_engine.local_image_catalog.validate_import', {}),
        ):
            mock = patch(target, **options)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(self.manager, '_child', side_effect=self.mock_smoke)
        self.smoke = mock.start()
        self.addCleanup(mock.stop)

    def _stop(self):
        if self.manager.thread is not None and self.manager.thread.is_alive():
            self.manager.cancel()
            self.manager.thread.join(5)

    def mock_smoke(self, machine, folder, name, args, **kwargs):
        self.assertEqual('smoke', name)
        self.assertEqual(b'model payload', (folder / self.target).read_bytes())
        self.assertIsNone(images.active_profile(self.engine, self.family))
        Path(args[-1]).write_bytes(b'test png' * 8)

    def select(self, name='原始 权重.safetensors', *, sha256=None):
        path = self.inputs / name
        path.write_bytes(b'model payload')
        row = local_model_copy.source_record(path, self.target, root=self.inputs, sha256=sha256)
        candidate = dict(label=name, kind='image', model=self.family, format='int8_convrot', source_kind='local',
                         files=[row], base_files=[], weights=self.target, bytes=row['bytes'], copy_bytes=row['bytes'],
                         required_bytes=row['bytes'], compatible=True)
        return path, self.manager._remember_local(candidate)

    def finish(self, phase):
        self.manager.thread.join(10)
        self.assertFalse(self.manager.thread.is_alive())
        task = self.manager.status()['task']
        self.assertEqual(phase, task['phase'], task.get('error'))
        self.assertFalse(list((self.engine / 'prepared/local-staging').glob('*')))
        return task

    def test_copy_publishes_only_after_smoke_and_survives_source_removal(self):
        source, selection = self.select()
        with self.assertRaisesRegex(ValueError, 'Inspect the Hugging Face'):
            self.manager.start(selection['id'])
        task = self.manager.start(selection['id'], local=True)
        self.assertEqual('copying', task['phase'])
        self.assertEqual({'copy_bytes', 'source_kind'}, selection.keys() & {'source', 'root', 'copy_bytes', 'source_kind'})
        result = self.finish('ready')
        installed = result['id']
        self.assertNotEqual(installed, selection['id'])
        self.assertEqual(selection['id'], result['selection_id'])
        source.rename(source.with_suffix('.moved'))
        self.manager.activate(installed)
        reopened = images.ImageModels(self.engine)
        active = reopened.status()['active'][self.family]
        self.assertEqual(installed, active['id'])
        pack = self.manager._profile_folder(installed)
        self.assertEqual(b'model payload', (pack / self.target).read_bytes())
        self.assertNotIn(str(self.inputs), (pack / 'profile.json').read_text(encoding='utf-8'))
        self.assertEqual(images.BUILTIN + self.family, reopened.activate(images.BUILTIN + self.family)['active'][self.family]['id'])
        _, duplicate = self.select('renamed.safetensors')
        self.manager.start(duplicate['id'], local=True)
        self.assertEqual(installed, self.finish('ready')['id'])
        self.smoke.assert_called_once()

    def test_changed_source_and_checksum_fail_without_publishing(self):
        path, selection = self.select()
        path.write_bytes(b'changed model')
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
        self.manager.start(selection['id'], local=True)
        self.assertIn('changed', self.finish('failed')['error'])
        _, corrupt = self.select(sha256='0' * 64)
        self.manager.start(corrupt['id'], local=True)
        self.assertIn('SHA-256', self.finish('failed')['error'])
        self.smoke.assert_not_called()
        self.assertIsNone(images.active_profile(self.engine, self.family))

    def test_disk_preflight_and_cancellation_leave_reusable_selection(self):
        _, selection = self.select()
        with patch('freevideo_engine.disk_space.free_bytes', return_value=0):
            self.manager.start(selection['id'], local=True)
            self.assertIn('disk space', self.finish('failed')['error'])
        self.manager.queue_busy = lambda: True
        self.manager.start(selection['id'], local=True)
        deadline = time.monotonic() + 5
        while self.manager.status()['task']['phase'] != 'waiting_for_gpu' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual('waiting_for_gpu', self.manager.status()['task']['phase'])
        self.manager.cancel()
        self.finish('cancelled')
        self.smoke.assert_not_called()
        self.manager.queue_busy = lambda: False
        self.manager.start(selection['id'], local=True)
        self.finish('ready')


if __name__ == '__main__':
    unittest.main()
