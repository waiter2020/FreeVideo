"""CPU checks for the component selections consumed by video generation."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from freevideo_engine import (comfy_assets, fast_tier, hf_video_models, local_comfy_h3_catalog,
                              model_components, model_components_generation as generation, swap_lora)
from freevideo_engine.hf_model_jobs import HubModelJobs
from freevideo_engine.storage import fingerprint


def publish(root, slot, files):
    version = sha256(json.dumps(sorted(files)).encode()).hexdigest()[:32]
    folder = model_components.version_directory(root, slot, version)
    folder.mkdir(parents=True)
    records = {}
    for name, contents in files.items():
        path = folder / name
        path.write_bytes(contents)
        records[name] = {'stamp': fingerprint(path), 'sha256': sha256(contents).hexdigest()}
    (folder / model_components.RECEIPT).write_text(json.dumps(dict(
        schema_version=1, slot=slot, id=version, files=records)), encoding='utf-8')
    model_components.choose(root, slot, version)
    return folder


class GenerationComponentTests(TestCase):
    def test_vae_directory_resolves_active_video_and_audio_independently(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'machine.json').touch()
            base = root / 'models' / 'h3-base'
            selected = publish(root, generation.VIDEO_VAE, {'config.json': b'{}'})
            self.assertEqual(generation.vae_directory(base), selected)
            self.assertEqual(generation.vae_directory(base, audio=True), base / 'audio_vae')
            (selected / 'config.json').write_bytes(b'changed')
            with self.assertRaises(model_components.ComponentError):
                generation.vae_directory(base)

    def test_pi_h3_uses_the_selected_pack_paths_and_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'machine.json').touch()
            model_root = root / 'models'
            adapter = fast_tier.mode_spec('fast4')['adapter']
            selected = publish(root, generation.PI_SLOTS['fast4'], {
                adapter: b'new adapter', fast_tier.CONSTANTS: b'new constants'})
            with patch('freevideo_engine.paths.installed_model_root', return_value=model_root):
                spec = fast_tier.request(SimpleNamespace(fast_mode='fast4'))
            self.assertEqual(Path(spec['adapter']), selected / adapter)
            self.assertEqual(spec['adapter_sha256'], sha256(b'new adapter').hexdigest())
            self.assertEqual(spec['adapter_bytes'], len(b'new adapter'))
            self.assertEqual(spec['constants_sha256'], sha256(b'new constants').hexdigest())

    def test_swap_managed_path_replaces_the_registered_alias(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            filename = swap_lora.catalog()['name']
            selected = publish(root, generation.SWAP_LORA, {filename: b'new adapter'})
            with patch('freevideo_engine.comfy_bridge.installation_root', return_value=root):
                rows = comfy_assets.resolve_loras(json.dumps([{'name': filename}]), [], None)
            self.assertEqual(Path(rows[0]['path']), selected / filename)

    def test_vae_import_selects_only_runtime_files(self):
        names = {'config.json', 'one.safetensors', 'two.safetensors', 'README.md', 'others/alt.safetensors'}
        self.assertEqual(generation._vae_layout(names), {'config.json', 'one.safetensors', 'two.safetensors'})
        with self.assertRaisesRegex(ValueError, 'config.json'):
            generation._vae_layout({'one.safetensors'})

    def test_native_hub_inspection_pins_checkpoint_and_checks_companion_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = local_comfy_h3_catalog
            revision = 'a' * 40
            checkpoint = 'diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors'
            header = json.dumps({'weight': dict(dtype='F32', shape=[1], data_offsets=[0, 4])}).encode()

            def info(repo, requested, token):
                if repo == catalog.COMPANION_REPO:
                    self.assertEqual(requested, catalog.COMPANION_REVISION)
                    companions = {('text_encoders/' if role == 'encoder' else 'vae/') + asset[0]: asset[1:]
                                  for role, asset in catalog.DEPENDENCIES.items()}
                    return {}, companions
                return {'id': repo, 'sha': revision}, {checkpoint: (len(header) + 12, 'b' * 64)}

            def row(repo, revision, entries, file, target):
                size, checksum = entries[file]
                return dict(repo=repo, revision=revision, file=file, target=target, bytes=size, sha256=checksum)

            def read_range(_, start, stop, token):
                return len(header).to_bytes(8, 'little') if start == 0 else header

            with (patch('freevideo_engine.hf_video_catalog._info', side_effect=info),
                  patch('freevideo_engine.hf_video_catalog._row', side_effect=row),
                  patch('freevideo_engine.hf_image_catalog._range', side_effect=read_range),
                  patch.object(catalog, '_dependency_source', side_effect=FileNotFoundError),
                  patch.object(catalog, 'checkpoint_format', return_value='comfy_int8') as layout):
                candidate = catalog.inspect_hub('example/model', 'ref2va', root, {}, '')
            self.assertEqual(layout.call_args.args[0], {'weight': ('F32', [1])})
            self.assertEqual(candidate['revision'], revision)
            self.assertEqual(candidate['files'][0]['file'], checkpoint)
            self.assertEqual(set(candidate['remote_dependencies']), set(catalog.DEPENDENCIES))

    def test_repair_publication_rolls_back_on_activation_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, staged = root / 'installed', root / 'verified'
            original.mkdir()
            staged.mkdir()
            (original / 'weights').write_bytes(b'original')
            (staged / 'weights').write_bytes(b'candidate')
            original_update = SimpleNamespace()
            with patch.object(original_update, 'publish', side_effect=ValueError('activation failed'), create=True):
                with self.assertRaisesRegex(ValueError, 'activation failed'):
                    HubModelJobs(root)._replace_import(original, staged, original_update.publish)
            self.assertEqual((original / 'weights').read_bytes(), b'original')
            self.assertEqual((staged / 'weights').read_bytes(), b'candidate')

    def test_legacy_preview_retains_its_prepared_cache(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'FreeVideo' / 'date' / 'work' / 'video.artifacts'
            artifact.mkdir(parents=True)
            (artifact / 'first-pass.pt').touch()
            profile = root / 'owned-profile'
            (artifact / 'video.json').write_text(json.dumps({'cache': str(profile / 'cache')}), encoding='utf-8')
            folders = SimpleNamespace(get_output_directory=lambda: str(root))
            with patch.dict(sys.modules, {'folder_paths': folders}):
                self.assertTrue(hf_video_models._retained_profile('test', profile))
                self.assertFalse(hf_video_models._retained_profile('test', root / 'other-profile'))


if __name__ == '__main__':
    main()
