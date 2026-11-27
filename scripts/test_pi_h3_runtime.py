"""CPU checks for Pi-H3 mode selection, scheduling and admission."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from freevideo_engine import capabilities, comfy_bridge, comfy_capabilities, fast_pack, fast_tier, policy, request_assets, workflow_repairs
from freevideo_engine.chain_plan import PlanError
from freevideo_engine.chain_runner import ChainRunner
from freevideo_engine.effort_forecast import timing_record
from freevideo_engine.geometry import geometry
from freevideo_engine.long_paths import io_path


class PiH3Tests(TestCase):
    def test_preserves_fast2_schedule_and_uses_fast4_schedule(self):
        self.assertEqual(fast_tier.mode_spec('fast2')['raw_grid'], (.93, .93 - .93 / 1.6, 0.))
        raw = fast_tier.mode_spec('fast4')['raw_grid']
        for value, expected in zip(raw, (.96, .6933333333333333, .4266666666666667, .16, 0.), strict=True):
            self.assertAlmostEqual(value, expected)
        for mode, steps in (('fast2', 2), ('fast4', 4)):
            with self.subTest(mode=mode):
                self.assertEqual(fast_tier.bridge_steps(mode), (False, steps, 3))

    def test_selected_cli_mode_uses_only_its_adapter_and_shared_constants(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'pi-h3'
            root.mkdir()
            adapters = {mode: fast_tier.mode_spec(mode)['adapter'] for mode in fast_tier.MODES}
            (root / adapters['fast4']).write_bytes(b'four')
            (root / fast_tier.CONSTANTS).write_bytes(b'a')
            def record(name, content, **fields):
                return dict(file=f'pi-h3/{name}', bytes=len(content), sha256=sha256(content).hexdigest(), engine=True, **fields)
            catalog = dict(folder='pi-h3', files=[record(fast_tier.CONSTANTS, b'a'),
                record(adapters['fast2'], b'two', modes=['fast2']), record(adapters['fast4'], b'four', modes=['fast4'])])
            with (patch.object(fast_pack, 'catalog', return_value=catalog),
                  patch('freevideo_engine.paths.installed_model_root', return_value=root.parent)):
                args = SimpleNamespace(fast_mode='fast4', base_steps=8, two_pass=True)
                spec = fast_tier.request(args)
                self.assertEqual((args.two_pass, args.base_steps, args.refine_steps), (False, 4, 3))
                self.assertEqual(Path(spec['adapter']), io_path(root / adapters['fast4']))
                self.assertEqual(Path(spec['constants']), io_path(root / fast_tier.CONSTANTS))
                self.assertEqual(fast_tier.adapter_row(spec)['sha256'], fast_tier.mode_spec('fast4')['adapter_sha256'])
                with self.assertRaisesRegex(FileNotFoundError, adapters['fast2']):
                    fast_tier.files(root.parent)
                (root / adapters['fast4']).write_bytes(b'fake')
                with self.assertRaisesRegex(FileNotFoundError, adapters['fast4']):
                    comfy_bridge.require_fast_model(dict(model_root=root.parent, prepared_format='int8_convrot'), 'fast4', environ={})

    def test_both_modes_reject_incompatible_requests(self):
        for mode in fast_tier.MODES:
            for flags in (dict(preview=True), dict(refine_from='video.mp4'), dict(continue_from='video.mp4'),
                          dict(upscale_from='video.mp4'), dict(original_audio=True), dict(redo_sound=True),
                          dict(step_cache='fbcache'), dict(upscale_factor=1.5), dict(parts=2)):
                with self.subTest(mode=mode, flags=flags), self.assertRaisesRegex(ValueError, 'cannot be used'):
                    fast_tier.bridge_steps(mode, **flags)

    def test_editor_chain_rejects_multiple_blocks_before_persisting(self):
        payload = dict(node='1', prompt={'1': dict(class_type='FreeVideoGenerate', inputs=dict(seed=0, fast_mode='fast4'))},
                       blocks=[dict(id=name, text='test', frames=39) for name in ('a', 'b')])
        runner = ChainRunner.__new__(ChainRunner)
        with self.assertRaisesRegex(PlanError, 'video made in parts') as failure:
            runner.create(payload)
        self.assertEqual(failure.exception.code, 'fast_mode_incompatible')

    def test_editor_chain_checks_the_selected_pack_before_persisting(self):
        payload = dict(node='1', prompt={'1': dict(class_type='FreeVideoGenerate', inputs=dict(seed=0, fast_mode='fast4'))},
                       blocks=[dict(id='a', text='test', frames=39)])
        runner = ChainRunner.__new__(ChainRunner)
        machine = dict(model_root='models', prepared_format='int8_convrot')
        with (patch.object(comfy_bridge, 'installation', return_value=(Path('.'), machine)),
              patch.object(fast_tier, 'files', side_effect=FileNotFoundError('Corrupt adapter')) as files,
              self.assertRaisesRegex(PlanError, 'Corrupt adapter') as failure):
            runner.create(payload)
        self.assertEqual(failure.exception.code, 'fast_mode_files_missing')
        self.assertEqual(files.call_args.args[1], 'fast4')

    def test_workflow_repairs_preserve_each_valid_mode(self):
        for mode in (*fast_tier.MODES, 'none', ['2', 0]):
            inputs = dict(fast_mode=mode)
            self.assertEqual(workflow_repairs.repair_inputs(inputs), [])
            self.assertEqual(inputs['fast_mode'], mode)
        inputs = dict(fast_mode='')
        self.assertEqual(workflow_repairs.repair_inputs(inputs), ['fast_mode'])
        self.assertEqual(inputs['fast_mode'], 'none')

    def test_fast4_uses_dwd_memory_and_skips_sampling_assets(self):
        fast4 = dict(geometry(960, 544, seconds=5), fast_mode='fast4')
        fast2 = dict(fast4, fast_mode='fast2')
        self.assertGreater(policy.fast_host_bytes(fast4), 0)
        self.assertEqual(policy.fast_host_bytes(fast4), policy.fast_host_bytes(fast2))
        self.assertEqual(policy.decode_workspace(fast4), policy.pixel_head_bytes(fast4))
        sampled, assets = request_assets._sampling(Path('.'), {}, dict(width=960, height=544, seconds=5, fast_mode='fast4'))
        self.assertEqual((sampled['enabled'], sampled['total_steps'], assets), (False, 4, []))

    def test_other_gpu_usage_uses_selected_fast_mode(self):
        query = comfy_capabilities._usage_query(dict(width='480', height='288', fast_mode='fast4'))
        self.assertFalse(query['two_pass'])
        self.assertEqual(query['steps'], 4)
        self.assertEqual(query['fast_mode'], 'fast4')

    def test_capabilities_publish_both_modes_and_keep_the_old_fast2_field(self):
        gib = 2**30
        machine = dict(device_backend='cuda', prepared_format='int8_convrot', gpu_vendor='nvidia',
                       hardware=dict(gpu_name='NVIDIA GeForce RTX 5090', system='Windows', capability=(12, 0),
                                     ram_total=64 * gib, vram_total=32 * gib, cuda_version='13.0'))
        result = capabilities.build(machine, [], size=(480, 288))
        fast_modes = result['fast_modes']
        self.assertEqual(result['fast_mode'], fast_modes['fast2'])
        self.assertEqual([fast_modes[mode]['steps'] for mode in fast_tier.MODES], [2, 4])
        for fast in fast_modes.values():
            self.assertTrue(fast['listed'])
            self.assertFalse(fast['two_pass'])
            self.assertTrue(fast['max_frames'][0])
        machine['prepared_format'] = 'fp8'
        fast_modes = capabilities.build(machine, [], size=(480, 288))['fast_modes']
        self.assertTrue(all(fast['unlisted_reason'] == 'engine' for fast in fast_modes.values()))

    def test_timings_keep_pi_h3_separate_from_ordinary_four_step_runs(self):
        canvas = geometry(960, 544, seconds=5)
        report = dict(success=True, geometry=canvas, request_seconds=8., lora=dict(adapter_count=1),
                      profile=dict(policy=dict(hardware=dict(gpu_uuid='GPU-test', system='Linux'))),
                      video=dict(phase='complete', config=dict(steps=4, fast_tier='fast4'),
                                 step_seconds=[1.] * 4))
        row = timing_record(report, 'GPU-test', 'Linux', stage_details=False)
        self.assertEqual(row['fast_mode'], 'fast4')
        self.assertFalse(row['adapters'])
        request = dict(canvas, task='t2va', two_pass=False, base_steps=4, refine_steps=0)
        self.assertEqual(capabilities._whole_records([row], request, False), [])
        self.assertEqual(capabilities._whole_records([row], dict(request, fast_mode='fast4'), False), [row])


if __name__ == '__main__':
    main()
