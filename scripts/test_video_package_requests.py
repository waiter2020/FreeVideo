"""Exercise package selection through the video bridge, CLI and chain admission."""
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from freevideo_engine import comfy_bridge, fast_tier, model_components, model_packages
from freevideo_engine.chain_runner import ChainRunner
from freevideo_engine.model_components_generation import PI_SLOTS
from freevideo_engine.monitoring import save
from freevideo_engine.storage import fingerprint


class VideoPackageRequests(TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        self.root = Path(temporary) / 'engine'
        self.model_root = Path(temporary) / 'external-models'
        self.machine = dict(root=str(self.root), model_root=str(self.model_root), prepared_format='int8_convrot')
        save(self.root / 'machine.json', self.machine)
        self.mode = 'fast2'
        self.slot = PI_SLOTS[self.mode]
        self.adapter = fast_tier.mode_spec(self.mode)['adapter']
        self.enterContext(patch.dict(os.environ, {'FREEVIDEO_HOME': str(self.root),
                                                 'FREEVIDEO_MODEL_ROOT': str(self.model_root)}))

    def publish(self, identity):
        folder = model_components.version_directory(self.root, self.slot, identity)
        folder.mkdir(parents=True)
        files = {}
        for name in (self.adapter, fast_tier.CONSTANTS):
            path = folder / name
            contents = (name + identity).encode()
            path.write_bytes(contents)
            files[name] = dict(stamp=fingerprint(path), sha256=sha256(contents).hexdigest())
        save(folder / model_components.RECEIPT, dict(schema_version=1, slot=self.slot, id=identity, files=files))
        return folder

    def bind(self, package, **bindings):
        with patch('freevideo_engine.model_components_generation._validate_fast'):
            return model_packages.update(self.root, package, bindings=bindings, machine=self.machine)

    def test_external_model_root_and_explicit_package_share_runtime_snapshot(self):
        adapter, constants = 'a' * 32, 'b' * 32
        adapter_path, constants_path = self.publish(adapter), self.publish(constants)
        package = model_packages.create(self.root, self.slot, 'Mixed')['id']
        self.bind(package, adapter=adapter, constants=constants)
        opened = comfy_bridge.require_fast_model(self.machine, self.mode, package_id=package)
        self.assertEqual(opened['package_id'], package)
        self.assertEqual(opened['bindings'], dict(adapter=adapter, constants=constants))
        with patch.dict(os.environ, {fast_tier.SNAPSHOT_ENV: fast_tier.snapshot(opened)}):
            spec = fast_tier.request(SimpleNamespace(fast_mode=self.mode, video_package=package))
        self.assertEqual(Path(spec['adapter']), adapter_path / self.adapter)
        self.assertEqual(Path(spec['constants']), constants_path / fast_tier.CONSTANTS)
        report = model_components.video_snapshot(fast=spec)
        self.assertEqual((report[self.slot], report[f'{self.slot}:constants']), (adapter, constants))

    def test_rebind_after_cache_lookup_or_cli_preflight_is_detected(self):
        one, two = 'a' * 32, 'b' * 32
        self.publish(one)
        self.publish(two)
        package = model_packages.create(self.root, self.slot, 'Changeable')['id']
        self.bind(package, adapter=one, constants=one)
        args = SimpleNamespace(fast_mode=self.mode, video_package=package)
        spec = fast_tier.request(args)
        self.bind(package, adapter=two)
        with self.assertRaisesRegex(ValueError, 'package changed'):
            fast_tier.recheck(spec)
        with (patch.dict(os.environ, {fast_tier.SNAPSHOT_ENV: spec['snapshot']}),
              self.assertRaisesRegex(ValueError, 'package changed')):
            fast_tier.request(args)

    def test_chain_admission_retains_explicit_package(self):
        identity = 'a' * 32
        self.publish(identity)
        package = model_packages.create(self.root, self.slot, 'Chain')['id']
        self.bind(package, adapter=identity, constants=identity)
        self.assert_chain_package(dict(seed=0, fast_mode=self.mode, video_package=package), package)

    def test_legacy_chain_pins_its_fast_default_when_queued(self):
        identity = 'a' * 32
        self.publish(identity)
        model_components.choose(self.root, self.slot, identity)
        package = model_packages.load(self.root)['legacy_fast'][self.mode]
        self.assert_chain_package(dict(seed=0, fast_mode=self.mode), package)

    def assert_chain_package(self, inputs, package):
        payload = dict(node='1', prompt={'1': dict(class_type='FreeVideoGenerate', inputs=inputs)},
                       blocks=[dict(id='a', text='test', frames=39)])
        runner = ChainRunner.__new__(ChainRunner)
        with (patch.object(comfy_bridge, 'installation', return_value=(self.root, self.machine)),
              patch.object(comfy_bridge, 'require_fast_model', wraps=comfy_bridge.require_fast_model) as require,
              patch.object(runner, 'lock', create=True),
              patch.object(runner, '_check_parts', side_effect=RuntimeError('checked')) as check,
              self.assertRaisesRegex(RuntimeError, 'checked')):
            runner.create(payload)
        self.assertEqual(require.call_args.kwargs['package_id'], inputs.get('video_package'))
        record = check.call_args.args[0]
        self.assertEqual(record['snapshot']['prompt']['1']['inputs']['video_package'], package)
        opened = comfy_bridge.require_fast_model(self.machine, self.mode, package_id=package)
        self.assertEqual(record['fast_package_snapshot'], fast_tier.snapshot(opened))


if __name__ == '__main__':
    main()
