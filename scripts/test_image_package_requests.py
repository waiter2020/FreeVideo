"""Image packages survive admission and select the same weights in the worker."""
import atexit
import hashlib
from pathlib import Path
import queue
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from freevideo_engine import hf_image_models, image_components, model_packages
from freevideo_engine.image_gen import image_model, service, worker
from freevideo_engine.image_gen.failure import Context, FailureError
from freevideo_engine.monitoring import save
from freevideo_engine.storage import fingerprint


class ImagePackageRequestTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix='fv-image-package-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.machine = dict(root=str(self.root), model_root=str(self.root / 'models'), device_backend='cuda')
        save(self.root / 'machine.json', self.machine)
        self.images = service.Service(lambda: (self.root, self.machine), lambda: str(self.root),
                                      video_busy=lambda: False)
        self.addCleanup(atexit.unregister, self.images.close)
        self.addCleanup(self.images.close)
        self.enterContext(patch.object(model_packages, 'image_ready', return_value=True))
        self.enterContext(patch.object(self.images, 'preflight', return_value=None))
        self.enterContext(patch.object(self.images, 'unified_total', return_value=None))
        self.enterContext(patch.object(self.images, 'output_path', return_value=str(self.root / 'result.jpg')))
        self.family = image_model.QWEN
        self.request = dict(model=self.family, prompt='A mountain.', width=512, height=512)
        self.first = self.package('a', 'First')
        self.second = self.package('b', 'Second')
        model_packages.activate(self.root, self.first, machine=self.machine)

    def install(self, token, family, role):
        identity = token * 24
        folder = hf_image_models._folder(self.root, identity)
        folder.mkdir(parents=True)
        weights = folder / 'weights.safetensors'
        payload = identity.encode()
        weights.write_bytes(payload)
        save(folder / image_components.RECEIPT, dict(
            schema_version=1, weights=weights.name,
            model=dict(id=identity, label=role, model=family, component=role),
            hashes={weights.name: hashlib.sha256(payload).hexdigest()}, stamps={weights.name: fingerprint(weights)}))
        return identity, weights

    def package(self, token, name, *, type_id='qwen-image-2.1', family=image_model.QWEN, role='dit'):
        identity, _ = self.install(token, family, role)
        package = model_packages.create(self.root, type_id, name)['id']
        model_packages.update(self.root, package, bindings={role: identity}, machine=self.machine)
        return package

    def prepare(self, **patches):
        value = dict(self.request, **patches)
        return self.images.prepare('generate', value, Context('generate', value))[2]

    def test_explicit_package_reaches_worker_and_ignores_client_identity(self):
        normalized = self.images.request(dict(self.request, package=self.second))
        self.assertEqual(normalized['package'], self.second)
        frozen = self.prepare(package=self.second, package_identity='client supplied')
        self.assertEqual(frozen['package'], self.second)
        self.assertNotEqual(frozen['package_identity'], 'client supplied')
        _, _, components, actual_package = worker.runner_source(self.root, self.family, self.root,
                                                                frozen['package'], frozen['package_identity'])
        self.assertEqual(components['dit']['id'], 'b' * 24)
        self.assertEqual(actual_package, self.second)
        context = Context('generate', frozen)
        self.assertNotIn('package_id', context.report('succeeded', outcome='succeeded'))
        context.note_model(dict(package_id=actual_package, components=components))
        report = context.report('succeeded', outcome='succeeded')
        self.assertEqual(report['package_id'], self.second)
        self.assertEqual(report['components']['dit']['id'], 'b' * 24)
        self.assertEqual(report['components']['vae']['id'], image_components.builtin_id(self.family, 'vae'))
        self.assertEqual(model_packages.load(self.root)['active']['image'], self.first)
        for invalid in (12, [], '.', 'builtin:unknown-model'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, '^invalid_request$'):
                self.prepare(package=invalid)

    def test_compose_passes_the_explicit_package_after_building_the_prompt(self):
        value = dict(width=512, height=512, package=self.second)
        built = dict(prompt='The scene.', width=512, height=512)
        module = SimpleNamespace(build=Mock(return_value=built))
        with patch('freevideo_engine.image_gen.compose', module, create=True):
            _, _, prepared = self.images.prepare('compose', value, Context('compose', value))
        self.assertEqual(prepared['package'], self.second)
        self.assertEqual(prepared['model'], self.family)
        self.assertEqual(prepared['prompt'], built['prompt'])

    def enqueue(self, **patches):
        self.images._busy = True
        self.images.queue_thread = Mock(**{'is_alive.return_value': True})
        key = self.images.start('generate', dict(self.request, **patches))['job']
        self.images._busy = False
        return key

    def test_queued_default_keeps_its_package_after_settings_switch(self):
        key = self.enqueue()
        model_packages.activate(self.root, self.second, machine=self.machine)
        with patch.object(self.images, 'begin') as begin:
            self.images.wait_queued()
        self.assertEqual(begin.call_args.args[0], key)
        self.assertEqual(begin.call_args.args[3]['package'], self.first)

    def test_rebinding_is_rejected_before_queued_dispatch_and_worker_load(self):
        key = self.enqueue(package=self.second)
        _, original = self.images.pending[0]['package_snapshot']
        replacement, _ = self.install('c', self.family, 'dit')
        model_packages.update(self.root, self.second, bindings={'dit': replacement}, machine=self.machine)
        with patch.object(self.images, 'begin') as begin, patch.object(self.images, 'failed', return_value={'phase': 'failed'}) as failed:
            self.images.wait_queued()
        begin.assert_not_called()
        self.assertEqual(failed.call_args.args[0], 'model_component_unavailable')
        self.assertEqual(self.images.finished[key]['phase'], 'failed')
        with self.assertRaisesRegex(image_components.ComponentError, 'package changed'):
            worker.runner_source(self.root, self.family, self.root, self.second, original)

    def test_light_warm_reads_the_requested_package_and_conservative_warm_keeps_the_loaded_one(self):
        self.images.thread = Mock(**{'is_alive.return_value': True})
        self.images.loaded_model = self.family
        self.images.loaded_identity = self.prepare()['package_identity']
        rejected = self.images.warm(dict(model=self.family, package=self.second, level='light', mode='conservative'))
        self.assertEqual(rejected, dict(warm='skipped', reason='not_loaded'))
        self.images.warm(dict(model=self.family, package=self.second, level='light'))
        warm = self.images.requests.get_nowait()
        self.assertEqual(warm['package'], self.second)
        self.assertIn(str(hf_image_models._folder(self.root, 'b' * 24) / 'weights.safetensors'), warm['files'])
        older = dict(warm, level='full', package=self.first, package_identity=self.images.loaded_identity)
        inbox = SimpleNamespace(rows=queue.Queue())
        inbox.rows.put(warm)
        self.assertEqual(worker.latest(inbox, older), warm)

    def test_package_without_a_family_uses_its_own_family_for_validation(self):
        h3 = self.package('d', 'H3', type_id='minimax-h3-image', family=image_model.H3)
        prepared = self.prepare(model=None, package=h3)
        self.assertEqual(prepared['model'], image_model.H3)
        self.assertGreaterEqual(prepared['width'], 2048)
        status = {'phase': 'failed', 'error': 'model_component_unavailable', 'report_id': None, 'stage': 'start'}
        with patch.object(self.images, 'failed', return_value=status):
            with self.assertRaisesRegex(FailureError, '^model_component_unavailable$'):
                self.prepare(package=h3)

    def test_h3_warm_places_the_packages_encoder_instead_of_the_shared_video_encoder(self):
        family = image_model.H3
        h3 = self.package('d', 'Picture', type_id='minimax-h3-image', family=family, role='text_encoder')
        video_encoder, _ = self.install('e', family, 'text_encoder')
        model_packages.set_shared_encoder(self.root, video_encoder, self.machine)
        self.machine['comfy_root'] = str(self.root)
        prepared = self.prepare(model=family, package=h3)
        module = SimpleNamespace(placed_encoder_bytes=Mock(return_value=0))
        policy = SimpleNamespace(gpu_budget_bytes=0, ram_budget_bytes=0, allocator_config='')
        with (patch.dict(sys.modules, {'freevideo_engine.image_gen.h3_pipeline': module}),
              patch.object(self.images, 'resident_endpoint', return_value=str(self.root / 'resident')),
              patch('freevideo_engine.hardware.detect', return_value=SimpleNamespace(system='test')),
              patch('freevideo_engine.kernel_capabilities.available_backends', return_value=[]),
              patch('freevideo_engine.policy.choose', return_value=policy),
              patch('freevideo_engine.idle_encoder.place_now', return_value={'state': 'ready'}) as place):
            self.images.place_encoder(self.machine, {}, None, h3, prepared['package_identity'])
        self.assertEqual(place.call_args.args[1]['encoder_component'], 'd' * 24)
        self.assertEqual(model_packages.shared_encoder_binding(self.root), video_encoder)

    def test_agent_prefers_settings_package_and_keeps_composer_override(self):
        info = dict(models=[dict(id=family, ready=True) for family in image_model.MODELS],
                    image_packages=[dict(active=True, ready=True, family=self.family)])
        self.assertEqual(image_model.choose(info), self.family)
        info['image_packages'][0]['family'] = image_model.H3
        self.assertEqual(image_model.choose(info), image_model.H3)
        self.assertEqual(image_model.choose(info, prefer=self.family), self.family)
        info['image_packages'][0]['ready'] = False
        self.assertEqual(image_model.choose(info), self.family)


if __name__ == '__main__':
    unittest.main()
