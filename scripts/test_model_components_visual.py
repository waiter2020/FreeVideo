import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from freevideo_engine import frame_preview, model_components, two_pass
from freevideo_engine.image_gen import step_preview
from freevideo_engine.model_components_visual import PROMPT_REQUIRED, prompt_layout
from freevideo_engine.monitoring import save
from freevideo_engine.prompt_vlm import assets
from freevideo_engine.storage import fingerprint


class VisualComponentsTests(unittest.TestCase):
    def select(self, root, slot, files, **metadata):
        identity = 'ab' * 16
        folder = model_components.version_directory(root, slot, identity)
        folder.mkdir(parents=True)
        records = {}
        for name, data in files.items():
            path = folder / name
            path.write_bytes(data)
            records[name] = dict(sha256=hashlib.sha256(data).hexdigest(), stamp=fingerprint(path))
        save(folder / model_components.RECEIPT,
             dict(schema_version=1, slot=slot, id=identity, files=records, **metadata))
        model_components.choose(root, slot, identity)
        return folder

    def test_preview_loaders_use_the_selected_weights(self):
        data = json.dumps(dict(version=1, channels=24, k=1, weight=[[0., 0., 0.]] * 24,
                               bias=[0.1, 0.2, 0.3])).encode()
        with TemporaryDirectory() as directory, patch.dict(os.environ, FREEVIDEO_HOME=directory):
            root = Path(directory)
            self.select(root, 'video-preview-linear', {frame_preview.WEIGHTS.name: data})
            self.select(root, 'h3-image-preview', {step_preview.WEIGHTS.name: data})
            self.addCleanup(frame_preview.clear)
            with patch.dict(os.environ, FREEVIDEO_FRAME_PREVIEW='1'):
                self.assertTrue(frame_preview.configure(root))
            self.assertEqual(frame_preview._weights[2], [0.1, 0.2, 0.3])
            self.assertEqual(step_preview.load_weights()[2], [0.1, 0.2, 0.3])

    def test_prompt_selection_keeps_builtin_install_status_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chosen = self.select(root, 'prompt-vlm', {'config.json': b'{}', 'model.safetensors': b'checkpoint'},
                                 repo='example/compatible-model', revision='stable', license='Apache-2.0')
            self.assertEqual(assets.model_directory(root), chosen)
            self.assertEqual(assets.model_details(root)['model'], 'example/compatible-model')
            self.assertTrue(assets.ready(root))
            self.assertFalse(assets.default_ready(root))

    def test_two_pass_plan_contains_the_selected_upscaler_digest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            name = Path(two_pass.UPSCALER['file']).name
            data = b'fixture'
            selected = self.select(root, 'h3-latent-upscaler', {name: data})
            plan = two_pass.plan(dict(width=1536, height=896, frames=97), installation_root=root)
            self.assertEqual(plan['upscaler_sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(two_pass.ensure_checkpoint(root=root), selected / name)

    def test_prompt_import_layout_accepts_single_and_indexed_weights(self):
        metadata = PROMPT_REQUIRED | {'tokenizer.json'}
        self.assertEqual(prompt_layout(metadata | {'model.safetensors', 'README.md'}),
                         metadata | {'model.safetensors'})
        indexed = metadata | {'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                              'model-00002-of-00002.safetensors'}
        self.assertEqual(prompt_layout(indexed), indexed)
        with self.assertRaises(ValueError):
            prompt_layout(indexed | {'model.safetensors'})


if __name__ == '__main__':
    unittest.main()
