"""Model packages: migration, bindings, locked parameters and the image lease."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from freevideo_engine import fast_pack, fast_tier, hf_image_models, image_components, model_components, model_packages
from freevideo_engine.image_gen import image_model
from freevideo_engine.locking import runtime_lock
from freevideo_engine.model_components_generation import PI_SLOTS
from freevideo_engine.monitoring import save
from freevideo_engine.storage import fingerprint


def publish(root, slot, files):
    version = hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()[:32]
    folder = model_components.version_directory(root, slot, version)
    folder.mkdir(parents=True)
    records = {}
    for name, contents in files.items():
        path = folder / name
        path.write_bytes(contents)
        records[name] = {'stamp': fingerprint(path), 'sha256': hashlib.sha256(contents).hexdigest()}
    (folder / model_components.RECEIPT).write_text(json.dumps(dict(
        schema_version=1, slot=slot, id=version, files=records)), encoding='utf-8')
    model_components.choose(root, slot, version)
    return folder


class ModelPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.machine = dict(root=str(self.root), model_root=str(self.root / 'models'), ready=True)
        save(self.root / 'machine.json', self.machine)

    def install(self, identity, family, role, contents=None):
        folder = hf_image_models._folder(self.root, identity)
        folder.mkdir(parents=True)
        weights = folder / 'weights.safetensors'
        payload = contents if contents is not None else identity.encode()
        weights.write_bytes(payload)
        save(folder / image_components.RECEIPT, dict(
            schema_version=1, weights=weights.name,
            model=dict(id=identity, label=role, model=family, component=role),
            hashes={weights.name: hashlib.sha256(payload).hexdigest()},
            stamps={weights.name: fingerprint(weights)}))
        return weights

    def activate(self, package_id):
        with patch.object(model_packages, 'image_ready', return_value=True):
            return model_packages.activate(self.root, package_id, machine=self.machine)

    def test_migration_keeps_image_paths_and_hashes(self):
        family = image_model.QWEN
        dit, encoder = 'a' * 24, 'b' * 24
        self.install(dit, family, 'dit')
        self.install(encoder, family, 'text_encoder')
        selections = {family: dict(dit=dit, text_encoder=encoder)}
        image_components.write_selections(self.root, selections)
        explicit = image_components.resolve(self.machine, family, selections=selections[family])
        migrated = image_components.resolve(self.machine, family)
        for role in image_components.ROLES:
            self.assertEqual(explicit['components'][role]['path'], migrated['components'][role]['path'])
            self.assertEqual(explicit['components'][role]['sha256'], migrated['components'][role]['sha256'])
        self.assertEqual(migrated['components']['vae']['builtin'], True)
        self.assertEqual(image_components.shared_encoder(self.machine), None)

    def test_two_qwen_packages_keep_their_own_denoisers(self):
        family = image_model.QWEN
        first, second = 'a' * 24, 'b' * 24
        self.install(first, family, 'dit', b'first denoiser')
        self.install(second, family, 'dit', b'second denoiser')
        one = model_packages.create(self.root, 'qwen-image-2.1', 'One')
        two = model_packages.create(self.root, 'qwen-image-2.1', 'Two')
        model_packages.update(self.root, one['id'], bindings={'dit': first}, machine=self.machine)
        model_packages.update(self.root, two['id'], bindings={'dit': second}, machine=self.machine)
        self.activate(one['id'])
        opened = image_components.resolve(self.machine, family, package=one['id'])
        other = image_components.resolve(self.machine, family, package=two['id'])
        self.assertEqual(opened['components']['dit']['id'], first)
        self.assertEqual(other['components']['dit']['id'], second)
        self.activate(two['id'])
        self.assertEqual(image_components.resolve(self.machine, family, package=one['id'])['components']['dit']['id'], first)
        self.assertEqual(model_packages.load(self.root)['active']['image'], two['id'])

    def test_builtin_role_ignores_an_installed_weight(self):
        family = image_model.QWEN
        dit = 'c' * 24
        weights = self.install(dit, family, 'dit')
        image_components.write_selections(self.root, {family: dict(dit=dit)})
        opened = image_components.resolve(self.machine, family, package='builtin:qwen-image-2.1')
        self.assertTrue(opened['components']['dit']['builtin'])
        self.assertNotEqual(Path(opened['components']['dit']['path']), weights)

    def test_builtin_and_active_image_packages_cannot_be_removed(self):
        package = model_packages.create(self.root, 'qwen-image-2.1', 'Kept')
        with self.assertRaisesRegex(model_packages.PackageError, 'built-in'):
            model_packages.remove(self.root, 'builtin:qwen-image-2.1')
        self.activate(package['id'])
        with self.assertRaisesRegex(model_packages.PackageError, 'in use'):
            model_packages.remove(self.root, package['id'])
        self.assertIn(package['id'], model_packages.load(self.root)['packages'])

    def test_qwen_encoder_cannot_bind_to_an_h3_package(self):
        encoder = 'd' * 24
        self.install(encoder, image_model.QWEN, 'text_encoder')
        package = model_packages.create(self.root, 'minimax-h3-image', 'H3')
        with self.assertRaisesRegex(model_packages.PackageError, 'different model component'):
            model_packages.update(self.root, package['id'], bindings={'text_encoder': encoder}, machine=self.machine)
        self.assertEqual(model_packages.load(self.root)['packages'][package['id']]['bindings']['text_encoder'], 'builtin')

    def test_pi_h3_steps_and_grid_stay_on_the_type(self):
        package = model_packages.create(self.root, 'pi-h3-2-step', 'Steps')
        kept = model_packages.update(self.root, package['id'], parameters={'steps': 2}, machine=self.machine)
        self.assertEqual(kept['parameters'], {})
        grid = list(fast_tier.mode_spec('fast2')['raw_grid'])
        grid[-1] = 0
        dropped = model_packages.update(self.root, package['id'], parameters={'grid': grid}, machine=self.machine)
        self.assertNotIn('grid', dropped['parameters'])
        with self.assertRaisesRegex(model_packages.PackageError, 'cannot be changed'):
            model_packages.update(self.root, package['id'], parameters={'steps': 3}, machine=self.machine)
        with self.assertRaisesRegex(model_packages.PackageError, 'Unknown'):
            model_packages.update(self.root, package['id'], parameters={'sigma': 1}, machine=self.machine)
        with self.assertRaisesRegex(model_packages.PackageError, 'cannot be changed'):
            model_packages.update(self.root, package['id'], parameters={'steps': True}, machine=self.machine)
        self.assertEqual(model_packages.load(self.root)['packages'][package['id']]['parameters'], {})

    def test_layout_mismatch_cannot_bind(self):
        adapter = fast_tier.mode_spec('fast4')['adapter']
        selected = publish(self.root, PI_SLOTS['fast4'], {adapter: b'new adapter', fast_tier.CONSTANTS: b'new constants'})
        package = model_packages.create(self.root, 'pi-h3-4-step', 'Replacement')
        with self.assertRaisesRegex(model_packages.PackageError, 'does not match this pi-H3 pack'):
            model_packages.update(self.root, package['id'], bindings={'adapter': selected.name}, machine=self.machine)
        self.assertEqual(model_packages.load(self.root)['packages'][package['id']]['bindings']['adapter'], 'builtin')

    def test_fast_mode_opens_the_migrated_adapter(self):
        adapter = fast_tier.mode_spec('fast2')['adapter']
        selected = publish(self.root, PI_SLOTS['fast2'], {adapter: b'old adapter', fast_tier.CONSTANTS: b'old constants'})
        model_root = self.root / 'models'
        with patch('freevideo_engine.paths.installed_model_root', return_value=model_root):
            spec = fast_tier.request(SimpleNamespace(fast_mode='fast2'))
        self.assertEqual(Path(spec['adapter']), selected / adapter)
        self.assertEqual(spec['adapter_sha256'], hashlib.sha256(b'old adapter').hexdigest())
        self.assertEqual(spec['constants_sha256'], hashlib.sha256(b'old constants').hexdigest())
        self.assertEqual(spec['steps'], 2)

    def test_a_changed_weight_fails_without_falling_back(self):
        family = image_model.QWEN
        encoder = 'e' * 24
        weights = self.install(encoder, family, 'text_encoder')
        package = model_packages.create(self.root, 'qwen-image-2.1', 'Encoder')
        model_packages.update(self.root, package['id'], bindings={'text_encoder': encoder}, machine=self.machine)
        self.activate(package['id'])
        weights.write_bytes(b'changed weight')
        with self.assertRaisesRegex(image_components.ComponentError, 'Encoder'):
            image_components.resolve(self.machine, family)
        self.assertEqual(model_packages.load(self.root)['packages'][package['id']]['bindings']['text_encoder'], encoder)

    def test_image_activation_waits_for_the_engine_lease(self):
        first = model_packages.create(self.root, 'moss-image-sft', 'First')
        second = model_packages.create(self.root, 'moss-image-sft', 'Second')
        self.activate(first['id'])
        with runtime_lock(self.root / 'engine.lock', inherit=False, wait_seconds=0):
            with self.assertRaisesRegex(model_packages.PackageError, 'image model package'):
                self.activate(second['id'])
        self.assertEqual(model_packages.load(self.root)['active']['image'], first['id'])

    def test_shared_encoder_does_not_change_an_image_package(self):
        picture, video = 'a' * 24, 'f' * 24
        self.install(picture, image_model.H3, 'text_encoder', b'picture encoder')
        self.install(video, image_model.H3, 'text_encoder', b'video encoder')
        package = model_packages.create(self.root, 'minimax-h3-image', 'Picture')
        model_packages.update(self.root, package['id'], bindings={'text_encoder': picture}, machine=self.machine)
        model_packages.set_shared_encoder(self.root, video, self.machine)
        self.assertEqual(model_packages.load(self.root)['packages'][package['id']]['bindings']['text_encoder'], picture)
        self.assertEqual(model_packages.shared_encoder_binding(self.root), video)
        model_packages.note_image_selection(self.root, image_model.H3, 'dit', None)
        self.assertEqual(model_packages.shared_encoder_binding(self.root), video)

    def test_unreadable_package_file_is_left_alone(self):
        path = self.root / model_packages.FILE
        path.write_text('{', encoding='utf-8')
        with self.assertRaisesRegex(model_packages.PackageError, 'could not be read'):
            model_packages.load(self.root)
        self.assertEqual(path.read_text(encoding='utf-8'), '{')

    def test_fast_lookup_without_an_installation_does_not_write_packages(self):
        (self.root / 'machine.json').unlink()
        if model_packages._bound_root(self.root / 'models', None) is not None:
            self.skipTest('an ancestor installation would be opened')
        with self.assertRaises(FileNotFoundError):
            fast_tier.files(self.root / 'models')
        self.assertFalse((self.root / model_packages.FILE).is_file())
        with self.assertRaises(model_packages.PackageError):
            fast_tier.files(self.root / 'models', package_id='builtin:pi-h3-2-step')

    def test_migration_preserves_external_family_preference_and_each_family_default(self):
        with TemporaryDirectory() as external:
            self.machine['model_root'] = external
            save(self.root / 'machine.json', self.machine)
            image_model.remember(external, image_model.QWEN)
            h3, moss = 'b' * 24, 'c' * 24
            self.install(h3, image_model.H3, 'dit')
            self.install(moss, image_model.MOSS, 'dit')
            image_components.write_selections(self.root, {image_model.H3: {'dit': h3}, image_model.MOSS: {'dit': moss}})
            document = model_packages.load(self.root)
            self.assertEqual(document['active']['image'], 'builtin:qwen-image-2.1')
            self.assertEqual(image_components.resolve(self.machine, image_model.H3)['components']['dit']['id'], h3)
            self.assertEqual(image_components.resolve(self.machine, image_model.MOSS)['components']['dit']['id'], moss)

    def test_v1_file_upgrades_without_changing_existing_package_ids(self):
        family = image_model.MOSS
        weight = 'd' * 24
        self.install(weight, family, 'dit')
        image_components.write_selections(self.root, {family: {'dit': weight}})
        document = model_packages.load(self.root)
        existing = document['image_defaults'][family]
        document['active']['image'] = 'builtin:qwen-image-2.1'
        document.pop('image_defaults')
        document['schema_version'] = 1
        save(self.root / model_packages.FILE, document)
        upgraded = model_packages.load(self.root)
        self.assertEqual(upgraded['schema_version'], model_packages.VERSION)
        self.assertEqual(upgraded['image_defaults'][family], existing)
        self.assertEqual(upgraded['active']['image'], 'builtin:qwen-image-2.1')
        self.assertEqual(json.loads((self.root / model_packages.FILE).read_text())['schema_version'], model_packages.VERSION)

    def test_missing_image_assets_do_not_show_as_ready_or_activate(self):
        package = model_packages.create(self.root, 'qwen-image-2.1')
        rows = {row['id']: row for row in model_packages.image_catalog(self.machine)}
        self.assertFalse(rows[package['id']]['ready'])
        self.assertTrue(rows[package['id']]['components_ready'])
        self.assertFalse(any(row['ready'] for row in rows.values()))
        with self.assertRaisesRegex(model_packages.PackageError, 'Install the built-in image model'):
            model_packages.activate(self.root, package['id'], machine=self.machine)

    def test_bad_package_type_and_saved_ids_raise_package_error(self):
        with self.assertRaisesRegex(model_packages.PackageError, 'supported model package type'):
            model_packages.perform(self.root, {'action': 'create', 'type': []}, self.machine)
        document = model_packages.load(self.root)
        document['active']['image'] = []
        save(self.root / model_packages.FILE, document)
        with self.assertRaisesRegex(model_packages.PackageError, 'could not be read'):
            model_packages.load(self.root)

    def test_fast_default_and_official_status_are_independent(self):
        mode = 'fast2'
        adapter = fast_tier.mode_spec(mode)['adapter']
        files = {adapter: b'adapter', fast_tier.CONSTANTS: b'constants'}
        model_packages.load(self.root)
        selected = publish(self.root, PI_SLOTS[mode], files)
        model_packages.note_fast_selection(self.root, mode, selected.name)
        with TemporaryDirectory() as external:
            installer = fast_pack.Installer(Path(external), self.root, machine=self.machine)
            self.assertTrue(installer.status(mode)['present'])
            self.assertFalse(installer.status(mode, target='builtin')['present'])
            opened = model_packages.resolve_fast(Path(external), mode, engine_root=self.root)
            self.assertEqual(opened['bindings'], dict.fromkeys(model_packages.FAST_ROLES, selected.name))
            self.assertEqual(opened['paths'][adapter], selected / adapter)
            self.assertEqual(opened['adapter_sha256'], hashlib.sha256(files[adapter]).hexdigest())
        model_packages.note_fast_selection(self.root, mode, None)
        self.assertEqual(model_packages.fast_default_bindings(self.root, mode), dict.fromkeys(model_packages.FAST_ROLES, 'builtin'))

    def test_mixed_fast_package_checks_only_its_official_roles(self):
        mode = 'fast2'
        adapter = fast_tier.mode_spec(mode)['adapter']
        selected = publish(self.root, PI_SLOTS[mode], {adapter: b'adapter', fast_tier.CONSTANTS: b'imported constants'})
        package = model_packages.create(self.root, 'pi-h3-2-step')
        with patch('freevideo_engine.model_components_generation._validate_fast'):
            model_packages.update(self.root, package['id'], bindings={'adapter': selected.name}, machine=self.machine)
        model_root = Path(self.machine['model_root'])
        official = model_root / fast_tier.FOLDER / fast_tier.CONSTANTS
        with patch.object(fast_pack, 'default_files', return_value={f'{fast_tier.FOLDER}/{official.name}': official}) as files:
            resolved = model_packages.resolve_fast(model_root, mode, package_id=package['id'])
        files.assert_called_once_with(model_root, mode=mode, names={fast_tier.CONSTANTS})
        self.assertEqual(resolved['paths'][adapter], selected / adapter)
        self.assertEqual(resolved['paths'][fast_tier.CONSTANTS], official)
        self.assertEqual(resolved['constants_sha256'], fast_tier.CONSTANTS_FILE[1])

    def test_video_binding_update_cannot_race_an_engine_request(self):
        package = model_packages.create(self.root, 'pi-h3-2-step')
        with runtime_lock(self.root / 'engine.lock', inherit=False, wait_seconds=0):
            with self.assertRaisesRegex(model_packages.PackageError, 'using this model'):
                model_packages.update(self.root, package['id'], bindings={'adapter': 'builtin'}, machine=self.machine)
            with self.assertRaisesRegex(model_packages.PackageError, 'using this model'):
                model_packages.remove(self.root, package['id'])

    def test_removal_protects_a_weight_used_only_by_an_inactive_image_package(self):
        family = image_model.QWEN
        identity = 'c' * 24
        weight = self.install(identity, family, 'dit')
        package = model_packages.create(self.root, 'qwen-image-2.1')
        model_packages.update(self.root, package['id'], bindings={'dit': identity}, machine=self.machine)
        manager = hf_image_models.ImageModels(self.root)
        with patch.object(manager, '_maintenance_lease', return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, 'every model package'):
                manager._remove_version(identity, family, 'dit')
        self.assertTrue(weight.is_file())

    def test_removal_protects_weights_used_only_by_an_inactive_video_package(self):
        from freevideo_engine.model_component_jobs import ComponentJobs

        mode, type_id = 'fast2', 'pi-h3-2-step'
        adapter = fast_tier.mode_spec(mode)['adapter']
        model_packages.load(self.root)
        selected = publish(self.root, PI_SLOTS[mode], {adapter: b'adapter', fast_tier.CONSTANTS: b'constants'})
        model_components.choose(self.root, type_id, None)
        package = model_packages.create(self.root, type_id)
        with patch('freevideo_engine.model_components_generation._validate_fast'):
            model_packages.update(self.root, package['id'], bindings={'adapter': selected.name}, machine=self.machine)
        manager = ComponentJobs(self.root, self.machine)
        slot = model_components.Slot(type_id, 'video', 'pi-H3', 'pi-H3', (), {}, '')
        folder, receipt = model_components.receipt(self.root, type_id, selected.name)
        with self.assertRaisesRegex(ValueError, 'every pi-H3 package'):
            manager._remove(slot, folder, receipt)
        self.assertTrue((selected / adapter).is_file())


if __name__ == '__main__':
    unittest.main()
