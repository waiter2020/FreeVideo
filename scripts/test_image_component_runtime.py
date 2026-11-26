"""CPU checks for component dispatch, warm-up and output provenance."""
import io
from importlib.util import find_spec
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from freevideo_engine.image_gen import image_model, prompt_cache, worker
from freevideo_engine.image_gen.failure import Context, component_facts
from freevideo_engine.image_gen.service import Service


def component(identity, path):
    return dict(id=identity, label=Path(path).name, path=str(path), sha256=identity[0] * 64)


class ImageComponentRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pack = self.root / 'images'
        (self.root / 'machine.json').write_text('{"model_root": "test"}', encoding='utf-8')

    def test_worker_changes_only_on_component_identity_and_keeps_reference_and_mask(self):
        family = image_model.QWEN
        stock = dict(dit=component('a-dit', self.pack / 'dit'),
                     text_encoder=component('b-text', self.pack / 'stock-encoder'),
                     vae=component('c-vae', self.pack / 'vae'))
        selected = dict(stock, text_encoder=component('d-text', self.root / 'replacement-encoder'))
        changed = dict(selected, vae=component('e-vae', self.root / 'replacement-vae'))

        def resolve(machine, model):
            self.assertEqual(machine['root'], str(self.root))
            self.assertEqual(model, family)
            components = next(selections)
            identity = tuple((item['id'], item['sha256']) for item in components.values())
            return dict(model=model, pack=str(self.pack), components=components, identity=identity)

        def request(number, reference=False, mask=False):
            target = self.root / f'{number}.png'
            return dict(model=family, prompt='test', width=256, height=256, path=str(target),
                        images=['uploaded-reference'] if reference else [], **({'mask': 'uploaded-mask'} if mask else {}))

        def factory(pack, device, *, progress, components):
            self.assertEqual(pack, self.pack)
            roles = tuple(item['id'] for item in components.values())
            events.append(('load', roles))

            def generate(_prompt, *, reference_images, edit_mask=None, **_):
                events.append(('generate', roles, reference_images, edit_mask))
                return SimpleNamespace(save=lambda stream, **_: stream.write(b'fixture'))

            return SimpleNamespace(generate=generate, close=lambda: events.append(('close', roles)), stats=dict)

        events, picture, mask = [], object(), SimpleNamespace(load=lambda: None)
        selections = iter((stock, selected, selected, changed))
        inbox = SimpleNamespace(closed=Event(), cancel=Event(), mutex=Lock())
        output = io.StringIO()
        with (patch('freevideo_engine.image_components.resolve', side_effect=resolve),
              patch.object(worker, 'Inbox', return_value=inbox),
              patch.object(worker, 'next_request', side_effect=(request(1), request(2, True),
                           request(3, True, True), request(4, True), None)),
              patch.object(worker, 'lease', side_effect=nullcontext),
              patch.object(worker, 'load_references', side_effect=lambda _, paths, **opts: [picture] if paths else []),
              patch.object(worker, 'drop', side_effect=lambda runner: runner.close()),
              patch.object(worker, 'release_memory'),
              patch.object(worker, 'attention_report', return_value={}),
              patch.object(worker, 'idle_wait', return_value=(1, None)),
              patch.dict('sys.modules', {'PIL': SimpleNamespace(Image=SimpleNamespace(open=lambda _: nullcontext(mask)))})):
            worker.serve(self.pack, engine_root=self.root, factory=factory, output=output)

        self.assertEqual([row[1] for row in events if row[0] == 'load'], [
            ('a-dit', 'b-text', 'c-vae'), ('a-dit', 'd-text', 'c-vae'), ('a-dit', 'd-text', 'e-vae')])
        self.assertEqual([row[2:] for row in events if row[0] == 'generate'],
                         [([], None), ([picture], None), ([picture], mask), ([picture], None)])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['phase'] for row in rows], ['model', 'result'] * 4)
        self.assertEqual(rows[4]['components'], component_facts(selected))
        self.assertNotIn(str(self.root), json.dumps(rows[4]))

    def test_read_ahead_uses_selected_h3_weights_and_leaves_shared_encoder_to_placement(self):
        paths = {role: self.root / f'{role}.safetensors' for role in ('dit', 'text_encoder', 'vae')}
        for path in paths.values():
            path.touch()
        components = {role: component('a-' + role, path) for role, path in paths.items()}
        with patch('freevideo_engine.image_components.resolve', return_value={'components': components}):
            self.assertEqual(Service.model_files({}, image_model.H3), [str(paths['dit']), str(paths['vae'])])
            self.assertEqual(Service.model_files({}, image_model.QWEN), list(map(str, paths.values())))

    def test_reference_conditioning_is_not_reused_after_encoder_switch(self):
        options = dict(encoder='same-name', comfy_root='comfy', comfy_python='py', model_paths='paths.yml')
        stock, selected = object(), object()
        with (patch.dict(prompt_cache._OPENED, clear=True),
              patch.object(prompt_cache, '_open_cache', side_effect=[stock, selected]) as open_cache):
            self.assertIs(prompt_cache.open_cache(**options, encoder_component='builtin'), stock)
            self.assertIs(prompt_cache.open_cache(**options, encoder_component='new'), selected)
            self.assertIs(prompt_cache.open_cache(**options, encoder_component='builtin'), stock)
        self.assertEqual(open_cache.call_count, 2)

    @unittest.skipUnless(find_spec('torch'), 'requires the installed image runtime')
    def test_h3_resident_encoder_retries_once_with_the_same_component_and_references(self):
        from freevideo_engine import encode
        from freevideo_engine.image_gen import h3_pipeline

        for cleanup, confirmed, failure_again in ((False, True, False), (False, True, True),
                                                  (True, True, False), (False, False, False)):
            with self.subTest(cleanup=cleanup, confirmed=confirmed, failure_again=failure_again):
                requests = []
                reference = SimpleNamespace(save=lambda path, **_: path.write_bytes(b'reference'))

                def run(args):
                    requests.append(args)
                    self.assertEqual(args.prompt_file.read_text(encoding='utf-8'), 'same prompt')
                    self.assertEqual(args.h3_image_references[0].read_bytes(), b'reference')
                    if len(requests) == 1:
                        args.out.with_suffix('.encoding.json').write_text(json.dumps({
                            'failure': {'kind': 'gpu_oom'}, 'failure_cleanup': {'complete': cleanup},
                        }), encoding='utf-8')
                        error = RuntimeError('CUDA error: out of memory')
                        error.worker_stop_confirmed = confirmed
                        raise error
                    if failure_again:
                        raise MemoryError('RAM guard: Windows commit headroom exhausted')

                with (patch.object(encode, 'run', side_effect=run),
                      patch.object(h3_pipeline.torch, 'load', return_value='conditioning')):
                    call = lambda: h3_pipeline._encode_once('same prompt', 'encoder', None, None, None, 9, 18,
                        [reference], resident_endpoint='resident', release_gpu=False, encoder_component='candidate')
                    if confirmed and not cleanup and not failure_again:
                        self.assertEqual(call(), 'conditioning')
                    else:
                        with self.assertRaisesRegex((RuntimeError, MemoryError), 'out of memory|RAM guard'):
                            call()

                self.assertEqual(len(requests), 2 if confirmed and not cleanup else 1)
                self.assertEqual(requests[0].resident_endpoint, 'resident')
                self.assertEqual([x.encoder_component for x in requests], ['candidate'] * len(requests))
                if len(requests) == 2:
                    self.assertIsNone(requests[1].resident_endpoint)
                    self.assertIs(requests[1].readonly_registration, False)
                    self.assertEqual(requests[1].extra_reserve_gib, 1.0)
                    self.assertNotEqual(requests[0].out, requests[1].out)

    def test_report_contains_effective_roles_and_omits_private_paths(self):
        path = self.root / 'candidate.safetensors'
        context = Context('generate', {'model': image_model.QWEN, 'images': ['reference']})
        context.components = component_facts({'dit': component('a-dit', path)})
        report = context.report('succeeded', outcome='succeeded')
        self.assertEqual(report['request']['references'], 1)
        self.assertEqual(report['components']['dit']['label'], path.name)
        self.assertNotIn(str(path), json.dumps(report))


if __name__ == '__main__':
    unittest.main()
