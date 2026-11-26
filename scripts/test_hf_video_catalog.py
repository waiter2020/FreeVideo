"""Focused fixture tests for public Hugging Face video discovery and inspection."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from freevideo_engine import hf_video_catalog as catalog


REPO = 'author/h3'
REVISION = 'a' * 40
SHA256 = 'b' * 64
NVIDIA = {'hardware': {'capability': (8, 9)}}


def file_row(name, data):
    prefix = f'blob {len(data)}\x00'.encode()
    return dict(rfilename=name, size=len(data), blobId=hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest())


def weights(name, size=1024):
    return dict(rfilename=name, size=size, lfs=dict(size=size, sha256=SHA256))


def manifest(slim=False):
    groups = ['root', *(f'blocks/{n:02d}' for n in range(catalog.BLOCKS))]
    if not slim:
        groups.extend(f'adaln/{n:02d}' for n in range(catalog.BLOCKS))
    return dict(source_id=SHA256, precision='fp8', scale_granularity='rowwise',
                groups=[dict(group=group, file=group + '.safetensors', bytes=1024, sha256=SHA256) for group in groups],
                linears={f'dense.{n}': dict(weight_shape=[256, 256], scale_shape=[1, 256], input_dtype='bfloat16')
                         for n in range(363)})


class CatalogTests(unittest.TestCase):
    def test_reference_normalizes_repo_and_rejects_paths_outside_hub(self):
        self.assertEqual(catalog.parse_reference('  ' + REPO + '  '), (REPO, None, None))
        self.assertEqual(catalog.parse_reference(f'https://huggingface.co/{REPO}/tree/v1/rowwise/cache'),
                         (REPO, 'v1', 'rowwise/cache'))
        for value in ('http://huggingface.co/author/h3', 'https://huggingface.co.invalid/author/h3',
                      f'https://huggingface.co/{REPO}/tree/main/%2E%2E',
                      f'https://huggingface.co/{REPO}/tree/main/cache/COM1', f'{REPO}@refs/heads/main'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                catalog.parse_reference(value)

    def test_search_omits_private_or_gated_results(self):
        data = json.dumps([dict(id=REPO, cardData={'license': 'mit'}), dict(id='other/gated', gated='auto'),
                           dict(id='other/private', private=True)]).encode()
        with patch.object(catalog, '_request', return_value=data) as request:
            self.assertEqual(catalog.search('mini max'), [{'repo': REPO, 'license': 'mit'}])
            self.assertIn('search=mini+max', request.call_args.args[0])

    def test_complete_fp8_manifest_pins_metadata_and_supports_gpu_policy(self):
        value = manifest()
        data = json.dumps(value).encode()
        prefix = 'rowwise/cache'
        name = f'{prefix}/manifest.json'
        siblings = [file_row(name, data), *(weights(f'{prefix}/{row["file"]}') for row in value['groups'])]
        info = dict(id=REPO, sha=REVISION, siblings=siblings, cardData={'license': 'mit'})
        def retrieve(path, **unused):
            return json.dumps(info).encode() if path.startswith('/api/') else data
        with TemporaryDirectory() as directory, patch.object(catalog, '_request', side_effect=retrieve):
            root = Path(directory)
            result = catalog.inspect(REPO, root, NVIDIA)
            self.assertTrue(result['compatible'])
            self.assertEqual(result['subfolder'], prefix)
            self.assertEqual(result['files'][0]['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(result['files'][1]['revision'], REVISION)
            self.assertEqual(result['files'][1]['target'], 'cache/root.safetensors')
            folder = root / 'prepared' / 'huggingface' / result['id']
            model_cache = folder / 'cache'
            model_cache.mkdir(parents=True)
            (model_cache / 'manifest.json').write_bytes(data)
            catalog.validate_import(result, folder, NVIDIA)
            (model_cache / 'manifest.json').write_bytes(data + b' ')
            with self.assertRaisesRegex(ValueError, 'changed after inspection'):
                catalog.validate_import(result, folder, NVIDIA)
            mac = catalog.inspect(REPO, root, {'device_backend': 'mps'})
            self.assertFalse(mac['compatible'])
            self.assertIn('Apple Silicon', mac['compatibility_reason'])

    def test_manifest_rejects_missing_layer_and_non_convrot_int8(self):
        incomplete = manifest()
        incomplete['groups'].pop()
        with self.assertRaisesRegex(ValueError, 'complete 50-block'):
            catalog._prepared_layout(incomplete)
        unsupported = dict(manifest(), precision='int8', format='freevideo-int8-slim-v1', rotation={})
        with self.assertRaises(ValueError):
            catalog._prepared_layout(unsupported)
        with self.assertRaisesRegex(ValueError, 'embedded LoRA'):
            catalog._prepared_layout(dict(manifest(), online_lora={'blocks': {'root': {'file': 'third-party'}}}))

    def test_raw_requires_compatible_config_and_complete_shards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, checkpoint = root / 'h3-base', root / catalog.LAYOUT
            (base / 'transformer').mkdir(parents=True)
            checkpoint.mkdir()
            config = b'{"num_layers":50}'
            spec = b'{"format_version":2}'
            (base / 'transformer' / 'config.json').write_bytes(config)
            (checkpoint / 'model_spec.json').write_bytes(spec)
            machine = dict(NVIDIA, base=str(base), checkpoint=str(checkpoint))
            base_prefix = 'h3-base/transformer'
            stage = catalog.LAYOUT
            index = b'{"weight_map":{"module.weight":"diffusion_pytorch_model-01.safetensors"}}'
            files = {base_prefix + '/config.json': config, stage + '/model_spec.json': spec,
                     base_prefix + '/diffusion_pytorch_model.safetensors.index.json': index}
            externals = [base_prefix + '/diffusion_pytorch_model-01.safetensors', stage + '/linear_branch/model.safetensors',
                         stage + '/adapters/default/adapter_model.safetensors', stage + '/adapters/turbo/adapter_model.safetensors']
            for path in (stage + '/linear_branch/config.json', stage + '/adapters/default/adapter_config.json',
                         stage + '/adapters/turbo/adapter_config.json'):
                files[path] = (b'{"type":"lora","version":1,"config":{"rank":64,"alpha":64,"targets":["test"]}}'
                               if 'adapter_config' in path else b'{}')
                (root / path).parent.mkdir(exist_ok=True, parents=True)
                (root / path).write_bytes(files[path])
            def retrieve(path, **unused):
                if path.startswith('/api/'):
                    rows = [file_row(name, body) for name, body in files.items()] + [weights(name) for name in externals]
                    return json.dumps(dict(id=REPO, sha=REVISION, siblings=rows)).encode()
                return files[path.split('/raw/' + REVISION + '/')[1]]
            with patch.object(catalog, '_request', side_effect=retrieve):
                result = catalog.inspect(REPO, root, machine)
                self.assertEqual(result['format'], 'raw')
                self.assertEqual(result['checkpoint'], catalog.LAYOUT)
                self.assertIsNone(result['cache'])
                externals.pop(0)
                with self.assertRaisesRegex(ValueError, 'missing or has an invalid size'):
                    catalog.inspect(REPO, root, machine)
                (checkpoint / 'model_spec.json').write_bytes(b'{"format_version":1}')
                with self.assertRaisesRegex(ValueError, 'installed VDN-H3 DMD8'):
                    catalog.inspect(REPO, root, machine)

    def test_public_gated_inspection_checks_access_with_read_token(self):
        response = dict(id=REPO, sha=REVISION, siblings=[], gated='manual')
        with patch.object(catalog, '_request', side_effect=[json.dumps(response).encode(), b'{}']) as request:
            catalog._info(REPO, None, 'hf_example')
            self.assertEqual(request.call_args.args[0], '/api/models/' + REPO + '/auth-check')
            self.assertEqual(request.call_args.kwargs, {'token': 'hf_example'})
        with patch.object(catalog, '_request', return_value=json.dumps(response).encode()):
            with self.assertRaisesRegex(ValueError, 'gated model'):
                catalog._info(REPO, None, '')
        response['private'] = True
        with patch.object(catalog, '_request', return_value=json.dumps(response).encode()), self.assertRaises(ValueError):
            catalog._info(REPO, None, 'hf_example')

    def test_adapter_scaling_targets_and_shapes(self):
        value = dict(type='lora', version=1, config=dict(rank=64, alpha=64, targets=['norm_out.linear'],
                     exact_targets=True, rank_pattern={'norm_out.linear': 16}, alpha_pattern={'norm_out.linear': 16}))
        config = catalog._adapter_config(value)
        actual = {'norm_out.linear.lora_A.turbo.weight': (16, 256), 'norm_out.linear.lora_B.turbo.weight': (512, 16)}
        catalog._adapter_shapes(actual, {'norm_out.linear': (512, 256)}, 'turbo', config)
        with self.assertRaisesRegex(ValueError, 'names or shapes'):
            catalog._adapter_shapes(actual, {'norm_out.linear': (512, 254)}, 'turbo', config)
        value['config']['alpha_pattern']['norm_out.linear'] = 32
        with self.assertRaisesRegex(ValueError, 'equal alpha and rank'):
            catalog._adapter_config(value)


if __name__ == '__main__':
    unittest.main()
