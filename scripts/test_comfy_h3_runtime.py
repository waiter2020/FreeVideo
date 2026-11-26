"""CPU checks for the native H3 prompt expansion within the FreeVideo queue."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from freevideo_engine import comfy_h3_runtime as runtime


class NativePromptTests(unittest.TestCase):
    def setUp(self):
        self.profile = {'id': 'installed-h3', 'preset': 'fl2va'}
        self.node = {'class_type': 'FreeVideoGenerate', 'inputs': {
            'text': 'A quiet beach', 'width': 1280, 'height': 768, 'seconds': 3., 'seed': 42,
        }}
        self.payload = {'prompt': {'17': self.node}, 'client_id': 'browser'}

    def lookup(self, identity):
        if identity is None or identity == self.profile['id']:
            return self.profile
        raise ValueError('Unknown H3 profile')

    def test_uses_active_profile_and_keeps_the_original_owner(self):
        original = deepcopy(self.payload)
        result = runtime.expand_prompt(self.payload, self.lookup)
        graph = result['prompt']
        self.assertEqual(self.payload, original)
        self.assertEqual(result['client_id'], 'browser')
        self.assertEqual(graph['17']['class_type'], 'FreeVideoGenerate')
        self.assertEqual(graph['17']['inputs'][runtime.PRESET], self.profile['id'])
        self.assertEqual(graph['17']['inputs']['base_steps'], 20)
        self.assertIs(graph['17']['inputs']['two_pass'], False)
        self.assertEqual(graph['17']['inputs'][runtime.VIDEO], ['freevideo_h3_17_video', 0])
        self.assertEqual(graph['freevideo_h3_17_condition']['inputs']['frames'], 73)
        self.assertEqual(graph['freevideo_h3_17_schedule']['inputs']['steps'], 20)

    def test_explicit_vdn_and_implicit_vdn_are_passthrough(self):
        self.assertIs(runtime.expand_prompt(self.payload, lambda identity: None), self.payload)
        self.node['inputs'][runtime.PRESET] = ''
        def must_not_look_up(identity):
            self.fail('The explicit VDN setting should not read the active profile')
        self.assertIs(runtime.expand_prompt(self.payload, must_not_look_up), self.payload)

    def test_requeue_rebuilds_owned_nodes_around_updated_values(self):
        expanded = runtime.expand_prompt(self.payload, self.lookup)
        inputs = expanded['prompt']['17']['inputs']
        inputs.update(seed=11, frames=56, base_steps=24, first=['44', 0])
        result = runtime.expand_prompt(expanded, self.lookup)['prompt']
        self.assertEqual(result['freevideo_h3_17_noise']['inputs']['noise_seed'], 11)
        self.assertEqual(result['freevideo_h3_17_condition']['inputs']['frames'], 56)
        self.assertEqual(result['freevideo_h3_17_condition']['inputs']['first'], ['44', 0])
        self.assertEqual(result['freevideo_h3_17_schedule']['inputs']['steps'], 24)

    def test_requeue_rebinds_the_shared_encoder_without_changing_the_already_queued_graph(self):
        self.profile['shared_encoder'] = {'id': 'old-encoder'}
        first = runtime.expand_prompt(self.payload, self.lookup)
        self.profile['shared_encoder'] = {'id': 'new-encoder'}
        second = runtime.expand_prompt(first, self.lookup)
        first_loader = first['prompt']['freevideo_h3_17_models']['inputs']
        second_loader = second['prompt']['freevideo_h3_17_models']['inputs']
        self.assertEqual(first_loader['encoder_component'], 'old-encoder')
        self.assertEqual(second_loader['encoder_component'], 'new-encoder')
        self.assertEqual(first_loader['preset'], second_loader['preset'])

    def test_vdn_features_are_rejected_at_the_owner_before_queueing(self):
        self.node['inputs']['preview'] = True
        self.node['inputs'][runtime.VIDEO] = ['unauthorized', 0]
        result = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertEqual(list(result), ['17'])
        self.assertNotIn(runtime.VIDEO, result['17']['inputs'])
        self.assertIn('VDN-specific', result['17']['inputs'][runtime.ERROR])

    def test_missing_profile_does_not_silently_run_vdn(self):
        self.node['inputs'][runtime.PRESET] = 'deleted-preset'
        result = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertEqual(result['17']['inputs'][runtime.ERROR], 'Unknown H3 profile')

    def test_static_media_mismatches_point_to_the_matching_preset(self):
        self.node['inputs']['references'] = ['42', 0]
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertIn('Use a Ref2VA preset', graph['17']['inputs'][runtime.ERROR])

        del self.node['inputs']['references']
        self.profile['preset'] = 'ref2va'
        self.node['inputs']['media'] = ['21', 0]
        self.payload['prompt']['21'] = {'class_type': 'FreeVideoMedia', 'inputs': {
            'assets': '[{"file":"input.png","role":"last","enabled":true}]',
        }}
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertIn('Use an FL2VA preset', graph['17']['inputs'][runtime.ERROR])

    def test_nondefault_upscale_is_rejected(self):
        self.node['inputs']['upscale_factor'] = '1.5'
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertIn('default upscale factor', graph['17']['inputs'][runtime.ERROR])

    def test_fastest_mode_requires_the_freevideo_model(self):
        self.node['inputs']['fast_mode'] = 'fast2'
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertEqual(list(graph), ['17'])
        self.assertIn('Turn off Fastest mode', graph['17']['inputs'][runtime.ERROR])
        self.assertIs(runtime.expand_prompt(self.payload, lambda identity: None), self.payload)

    def test_native_and_vdn_outputs_must_be_queued_separately(self):
        self.payload['prompt']['50'] = deepcopy(self.node)
        self.payload['prompt']['50']['inputs'][runtime.PRESET] = ''
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        self.assertEqual(set(graph), {'17', '50'})
        for node in graph.values():
            self.assertIn('separate prompts', node['inputs'][runtime.ERROR])
            self.assertNotIn(runtime.VIDEO, node['inputs'])

        self.payload['prompt']['50']['inputs'][runtime.PRESET] = self.profile['id']
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        for owner in ('17', '50'):
            self.assertEqual(graph[owner]['inputs'][runtime.VIDEO], [f'freevideo_h3_{owner}_video', 0])

    def test_editable_prompt_omits_internal_queue_nodes_and_links(self):
        expanded = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        editable = runtime._editable_prompt(expanded)
        self.assertEqual(list(editable), ['17'])
        self.assertNotIn(runtime.VIDEO, editable['17']['inputs'])
        self.assertEqual(editable['17']['inputs'][runtime.PRESET], self.profile['id'])

    def test_native_windows_loading_maps_source_read_only(self):
        header = b'{"weights":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}}'
        tensor = object()
        metadata = {'quantization': 'test'}
        loader = MagicMock()
        reader = loader.return_value.__enter__.return_value
        reader.metadata.return_value = metadata
        dtype = SimpleNamespace(itemsize=1)
        torch = SimpleNamespace(frombuffer=lambda view, dtype: SimpleNamespace(reshape=lambda shape: (tensor, bytes(view), shape)))
        modules = {'safetensors': SimpleNamespace(safe_open=loader),
                   'comfy.utils': SimpleNamespace(_TYPES={'I8': dtype}), 'torch': torch}
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / 'fixture.safetensors')
            Path(path).write_bytes(len(header).to_bytes(8, 'little') + header + b'ab')
            with (patch.dict('sys.modules', modules),
                  patch.object(runtime.mmap, 'mmap', wraps=runtime.mmap.mmap) as mapper):
                self.assertEqual(runtime._read_windows_weights(path), ({'weights': (tensor, b'ab', [2])}, metadata))
                self.assertEqual(mapper.call_args.kwargs['access'], runtime.mmap.ACCESS_READ)
            loader.assert_called_once_with(path, framework='pt', device='cpu', backend='pread')
            reader.get_tensor.assert_not_called()

    def test_sampling_releases_weights_before_both_decoders(self):
        graph = runtime.expand_prompt(self.payload, self.lookup)['prompt']
        handoff = 'freevideo_h3_17_release_sampling'
        self.assertEqual(graph[handoff]['class_type'], runtime.RELEASE_SAMPLER)
        self.assertEqual(graph[handoff]['inputs']['latent'], ['freevideo_h3_17_sample', 0])
        for decoder in ('images', 'audio'):
            self.assertEqual(graph[f'freevideo_h3_17_{decoder}']['inputs']['samples'], [handoff, 0])

    def test_smoke_is_deterministic_and_ref2va_includes_an_image(self):
        args = {'variant': 'ref2va', 'filename_prefix': 'FreeVideo/.model-smoke/snapshot'}
        graph = runtime.smoke_graph('installed-h3', **args)
        self.assertEqual(graph, runtime.smoke_graph('installed-h3', **args))
        self.assertEqual(graph['freevideo_h3_smoke_smoke_image']['class_type'], 'EmptyImage')
        self.assertEqual(graph['freevideo_h3_smoke_condition']['inputs']['frames'], 39)
        self.assertEqual(graph['freevideo_h3_smoke_schedule']['inputs']['steps'], 2)
        self.assertEqual(graph[runtime.SMOKE_OUTPUT_NODE]['inputs']['filename_prefix'], args['filename_prefix'])
        self.assertNotIn('owner', graph['freevideo_h3_smoke_models']['inputs'])
        keyframes = runtime.smoke_graph('installed-h3', variant='fl2va')
        self.assertEqual(keyframes['freevideo_h3_smoke_condition']['inputs']['first'],
                         ['freevideo_h3_smoke_smoke_image', 0])
        self.assertEqual(keyframes['freevideo_h3_smoke_condition']['inputs']['last'],
                         ['freevideo_h3_smoke_smoke_tail', 0])
        self.assertNotEqual(keyframes['freevideo_h3_smoke_smoke_image']['inputs']['color'],
                            keyframes['freevideo_h3_smoke_smoke_tail']['inputs']['color'])

    def test_progress_and_errors_stay_with_the_native_owner(self):
        progress = {'prompt_id': 'prompt-1', 'node': 'freevideo_h3_17_sample', 'value': 4, 'max': 20}
        failure = {'prompt_id': 'prompt-1', 'node_id': 'freevideo_h3_17_condition', 'exception_message': 'Failed'}
        server = object()
        with (patch.dict(runtime._ACTIVE, {'prompt-1': {'17'}}, clear=True),
              patch('freevideo_engine.comfy_progress.publish') as publish):
            self.assertIs(runtime._progress_event(server, 'progress', progress), progress)
            publish.assert_called_once_with(server, '17', {
                'label': 'Sampling', 'phase': 'sampling', 'stage': 'single', 'done': 4, 'total': 20, 'unit': 'step',
            })
            self.assertEqual(runtime._progress_event(server, 'execution_error', failure)['node_id'], '17')
            private = {**failure, 'prompt_id': 'another-prompt'}
            self.assertIs(runtime._progress_event(server, 'execution_error', private), private)


if __name__ == '__main__':
    unittest.main()
