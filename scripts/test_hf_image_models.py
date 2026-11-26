import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from freevideo_engine import hf_image_models as images
from freevideo_engine.monitoring import save


class ImageModelsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        save(self.root / 'machine.json', dict(ready=True, python='engine-python', comfy_python='comfy-python',
                                             model_root=str(self.root / 'models'), gpu_uuid='test-gpu'))
        self.manager = images.ImageModels(self.root)
        ready = patch('freevideo_engine.hf_image_models._ready', return_value=True)
        ready.start()
        self.addCleanup(ready.stop)

    def profile(self, identity, model):
        folder = images._folder(self.root, identity)
        relative = 'image_gen/denoiser.safetensors'
        weights = folder / relative
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'image model weights')
        save(folder / 'profile.json', dict(schema_version=1, gpu_uuid='test-gpu',
                                          model=dict(id=identity, model=model, label='Test ' + model),
                                          stamps={relative: images._stamp(weights)}))
        return weights

    def test_activation_persists_by_family_and_rolls_back_only_the_selected_family(self):
        qwen, h3 = images.image_model.QWEN, images.image_model.H3
        self.profile('a' * 24, qwen)
        self.profile('b' * 24, h3)
        self.manager.activate('a' * 24)
        self.manager.activate('b' * 24)
        reopened = images.ImageModels(self.root)
        self.assertEqual(reopened.status()['active'][qwen]['id'], 'a' * 24)
        result = reopened.activate(images.BUILTIN + qwen)
        self.assertEqual(result['active'][qwen]['id'], images.BUILTIN + qwen)
        self.assertEqual(result['active'][h3]['id'], 'b' * 24)
        self.assertEqual(images.active_profile(self.root, h3)['id'], 'b' * 24)

    def test_corrupt_selection_is_reimportable_and_generation_uses_the_builtin(self):
        family = images.image_model.QWEN
        weights = self.profile('a' * 24, family)
        self.manager.activate('a' * 24)
        weights.write_bytes(b'changed')
        self.assertIsNone(images.active_profile(self.root, family))
        self.assertFalse(self.manager._already_active('a' * 24))
        self.assertFalse(any(row['id'] == 'a' * 24 for row in self.manager.status()['installed']))

    def test_import_runs_validation_and_smoke_before_activation(self):
        family = images.image_model.QWEN
        identity = 'a' * 24
        folder = images._folder(self.root, identity)
        target = 'image_gen/denoiser.safetensors'
        weights = folder / target
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'candidate')
        license_data = b'license fixture'
        (folder / 'LICENSE').write_bytes(license_data)
        checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
        files = [dict(target=target, bytes=9, sha256=checksum),
                 dict(target='LICENSE', bytes=len(license_data), sha256=hashlib.sha256(license_data).hexdigest())]
        companion = images.image_pack.base(self.root / 'models') / 'vae' / 'decoder.safetensors'
        companion.parent.mkdir(parents=True)
        companion.write_bytes(b'official decoder')
        base_files = [dict(repo='test/official', file='vae/decoder.safetensors', bytes=companion.stat().st_size,
                           sha256=hashlib.sha256(companion.read_bytes()).hexdigest())]
        candidate = dict(id=identity, model=family, bytes=9 + len(license_data), files=files,
                         base_files=base_files, compatible=True)
        self.manager._remember(candidate)

        def child(machine, directory, name, args, **kwargs):
            self.assertIsNone(images.active_profile(self.root, family))
            if name == 'smoke':
                self.assertEqual('comfy-python', kwargs['python'])
                Path(args[-1]).write_bytes(b'PNG fixture' * 8)

        with (patch('freevideo_engine.hf_image_catalog.validate_import') as validate,
              patch.object(self.manager, '_child', side_effect=child)):
            self.manager.start(identity)
            self.manager.thread.join(5)
        validate.assert_called_once()
        self.assertEqual('ready', self.manager.status()['task']['phase'])
        self.assertEqual(companion.read_bytes(), (folder / 'image_gen' / base_files[0]['file']).read_bytes())
        self.assertIsNone(images.active_profile(self.root, family))
        self.manager.activate(identity)
        self.assertEqual(identity, images.active_profile(self.root, family)['id'])


if __name__ == '__main__':
    unittest.main()
