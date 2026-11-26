import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from freevideo_engine import hf_image_models, image_components
from freevideo_engine.image_gen import image_model
from freevideo_engine.monitoring import save
from freevideo_engine.storage import fingerprint


class ImageComponentActivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.machine = dict(root=str(self.root), model_root=str(self.root / 'models'), ready=True,
                            python='python', comfy_python='comfy-python', gpu_uuid='test')
        save(self.root / 'machine.json', self.machine)
        self.manager = hf_image_models.ImageModels(self.root)
        patcher = patch.object(hf_image_models, '_ready', return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def install(self, identity, family, role):
        folder = hf_image_models._folder(self.root, identity)
        folder.mkdir(parents=True)
        weights = folder / 'weights.safetensors'
        contents = identity.encode()
        weights.write_bytes(contents)
        save(folder / image_components.RECEIPT, dict(schema_version=1, weights=weights.name,
             model=dict(id=identity, label=role, model=family, component=role),
             hashes={weights.name: hashlib.sha256(contents).hexdigest()}, stamps={weights.name: fingerprint(weights)}))
        return weights

    def activate(self, identity, family, role):
        self.manager.activate(identity, model=family, component=role)
        self.manager.thread.join(5)
        self.assertFalse(self.manager.thread.is_alive())
        return self.manager.status()['task']['phase']

    def test_qwen_switch_preserves_other_components_and_verifies_requested_combination(self):
        family = image_model.QWEN
        dit, encoder, vae = ('a' * 24, 'b' * 24, 'c' * 24)
        for identity, role in ((dit, 'dit'), (encoder, 'text_encoder'), (vae, 'vae')):
            self.install(identity, family, role)
        image_components.write_selections(self.root, {family: dict(dit=dit, vae=vae)})
        with patch.object(self.manager, '_activate_picture') as smoke:
            self.assertEqual(self.activate(encoder, family, 'text_encoder'), 'ready')
        checked = smoke.call_args.args[3]['components']
        self.assertEqual({role: row['id'] for role, row in checked.items()}, dict(dit=dit, text_encoder=encoder, vae=vae))
        with patch.object(self.manager, '_activate_picture'):
            self.assertEqual(self.activate(image_components.builtin_id(family, 'text_encoder'), family, 'text_encoder'), 'ready')
        self.assertEqual(image_components.read_selections(self.root)[family], dict(dit=dit, text_encoder=None, vae=vae))

    def test_failed_validation_keeps_previous_selection(self):
        family = image_model.QWEN
        encoder = 'b' * 24
        self.install(encoder, family, 'text_encoder')
        with patch.object(self.manager, '_activate_picture', side_effect=ValueError('reference validation failed')):
            self.assertEqual(self.activate(encoder, family, 'text_encoder'), 'failed')
        self.assertEqual(image_components.read_selections(self.root), {})
        self.assertEqual(self.manager.status()['task']['error'], 'reference validation failed')

    def test_damaged_selected_checkpoint_is_visible_and_resolvable_only_after_explicit_reset(self):
        family = image_model.QWEN
        encoder = 'b' * 24
        file = self.install(encoder, family, 'text_encoder')
        image_components.write_selections(self.root, {family: dict(text_encoder=encoder)})
        file.write_bytes(b'changed weight')
        status = self.manager.status()['components'][family]['text_encoder']
        self.assertEqual(status['active']['id'], encoder)
        self.assertTrue(status['active']['damaged'])
        with self.assertRaisesRegex(image_components.ComponentError, 'Qwen-Image 2.1 Encoder'):
            image_components.resolve(self.machine, family)
        with patch.object(self.manager, '_activate_picture'):
            self.assertEqual(self.activate(image_components.builtin_id(family, 'text_encoder'), family, 'text_encoder'), 'ready')
        self.assertTrue(image_components.resolve(self.machine, family)['components']['text_encoder']['builtin'])

    def test_h3_encoder_resolution_uses_single_selection_and_allows_pinned_builtin(self):
        identity = 'e' * 24
        file = self.install(identity, image_model.H3, 'text_encoder')
        image_components.write_selections(self.root, {image_model.H3: dict(text_encoder=identity)})
        shared = image_components.shared_encoder(self.machine)
        image_encoder = image_components.resolve(self.machine, image_model.H3)['components']['text_encoder']
        self.assertEqual(shared, image_encoder)
        self.assertEqual(Path(shared['path']), file)
        self.assertIsNone(image_components.shared_encoder(self.machine, identity='builtin'))


if __name__ == '__main__':
    unittest.main()
