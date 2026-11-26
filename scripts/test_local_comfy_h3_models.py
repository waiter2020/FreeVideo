"""Local ComfyUI H3 profiles use owned, content-addressed assets."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from freevideo_engine import hf_video_models as models, local_comfy_h3_catalog as catalog


class LocalComfyH3ModelsTest(unittest.TestCase):
    def setUp(self):
        storage = TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        root = Path(storage.name)
        self.engine = root / 'engine'
        self.models = root / 'ComfyUI' / 'data' / 'diffusion_models' / 'minimax'
        self.models.mkdir(parents=True)
        self.weight = self.models / 'local_h3.safetensors'
        self.weight.write_bytes(b'local checkpoint')
        self.official = self.engine / 'prepared' / 'official'
        self.official.mkdir(parents=True)
        (self.official / 'manifest.json').write_text('{"precision":"int8"}', encoding='utf-8')
        encoder_root = self.engine / 'models' / 'encoder'
        self.machine = dict(ready=True, python='python', gpu_uuid='gpu', cache=str(self.official),
                            base='base', checkpoint='checkpoint', encoder_model_root=str(encoder_root))
        (self.engine / 'machine.json').write_text(json.dumps(self.machine), encoding='utf-8')
        dependencies = {}
        for role, folder in (('encoder', encoder_root / 'text_encoders'),
                             ('video_vae', self.models.parents[1] / 'vae'),
                             ('audio_vae', self.models.parents[1] / 'vae')):
            folder.mkdir(parents=True, exist_ok=True)
            name = role + '.safetensors'
            data = role.encode()
            (folder / name).write_bytes(data)
            dependencies[role] = (name, len(data), hashlib.sha256(data).hexdigest())
        self.encoder = encoder_root / 'text_encoders' / dependencies['encoder'][0]
        patched = patch.object(catalog, 'DEPENDENCIES', dependencies)
        patched.start()
        self.addCleanup(patched.stop)
        self.smoke_profiles = []
        self.manager = models.VideoModels(self.engine, comfy_smoke=self.smoke)
        self.manager._child = self.validate_video
        with patch.object(catalog, '_layout', return_value='comfy_int8'):
            [row] = self.manager.scan_comfy_h3(self.models)['candidates']
        self.selection = row['id']

    def smoke(self, profile, output, check):
        check()
        self.assertEqual(set(profile['assets']), {'checkpoint', 'encoder', 'video_vae', 'audio_vae'})
        self.assertTrue(all(Path(path).is_file() for path in profile['assets'].values()))
        self.assertTrue(Path(profile['checkpoint']).is_relative_to(self.engine / catalog.OWNED))
        self.assertNotIn(profile['id'], [item['id'] for item in self.manager.status()['installed']])
        self.smoke_profiles.append(profile)
        output.write_bytes(b'short clip with audio' * 6)
        return output

    def validate_video(self, machine, folder, name, args, **kwargs):
        self.assertEqual(name, 'smoke-validation')
        self.assertTrue(Path(args[-1]).is_file())

    def finish(self, preset):
        selected = self.manager.inspect_comfy_h3(self.selection, preset)
        self.manager._work_local(self.manager.pending[selected['id']])
        return selected, self.manager.status()['task']['candidate']

    def test_import_two_presets_reuses_all_assets_and_restores_official(self):
        first_selection, first_model = self.finish('fl2va')
        second_selection, second_model = self.finish('ref2va')
        vae_bytes = sum(catalog.DEPENDENCIES[role][1] for role in ('video_vae', 'audio_vae'))
        self.assertEqual(first_selection['copy_bytes'], self.weight.stat().st_size + vae_bytes)
        self.assertTrue(Path(self.smoke_profiles[0]['encoder']).samefile(self.encoder))
        self.assertEqual(second_selection['copy_bytes'], 0)
        self.assertNotEqual(first_model['id'], second_model['id'])
        self.assertEqual(self.smoke_profiles[0]['assets'], self.smoke_profiles[1]['assets'])
        self.assertNotIn('assets', second_model)
        self.assertNotIn(str(self.models), json.dumps(second_model))
        self.weight.unlink()
        self.manager._activate(second_model['id'])
        active = models.active_comfy_h3(self.engine)
        self.assertEqual(active['variant'], 'ref2va')
        self.assertEqual(Path(self.manager._machine()['cache']), self.official)
        self.assertEqual(models.resolve_comfy_h3(self.engine, first_model['id'])['variant'], 'fl2va')
        self.manager._activate(models.OFFICIAL_ID)
        self.assertIsNone(models.active_comfy_h3(self.engine))

    def test_failed_smoke_creates_no_installed_profile(self):
        selection = self.manager.inspect_comfy_h3(self.selection, 'fl2va')
        with patch.object(self.manager, 'comfy_smoke', side_effect=ValueError('Cannot generate the test clip.')):
            self.manager._work(self.manager.pending[selection['id']], '')
        status = self.manager.status()
        self.assertEqual(status['task']['phase'], 'failed')
        self.assertEqual(status['active']['id'], models.OFFICIAL_ID)
        self.assertEqual(len(status['installed']), 1)
        retry = self.manager.inspect_comfy_h3(self.selection, 'fl2va')
        self.assertEqual(retry['copy_bytes'], 0)

    def test_invalid_layout_still_reports_bytes(self):
        with patch.object(catalog, '_layout', side_effect=ValueError('Invalid layout')):
            [row] = self.manager.scan_comfy_h3(self.models)['candidates']
        self.assertFalse(row['compatible'])
        self.assertEqual(row['bytes'], self.weight.stat().st_size)


if __name__ == '__main__':
    unittest.main()
