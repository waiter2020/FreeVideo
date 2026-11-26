"""Local VDN-H3 layout fixtures use real, small safetensors headers."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from freevideo_engine import hf_video_catalog as catalog, local_video_catalog as local


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_tensors(path, tensors):
    header, size = {}, 0
    for name, dtype, shape in tensors:
        count = math.prod(shape) * local.TENSOR_BYTES[dtype]
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[size, size + count])
        size += count
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = len(encoded).to_bytes(8, 'little') + encoded + bytes(size)
    path.write_bytes(content)
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), tensors=len(header))


class LocalVideoCatalogTest(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.root = Path(storage.name)
        self.directory = self.root / '本地视频权重'
        self.directory.mkdir()
        self.machine = {'hardware': {'capability': (8, 9)}, 'base': str(self.root / 'installed' / catalog.RAW_BASE),
                        'checkpoint': str(self.root / 'installed' / catalog.LAYOUT)}

    def _raw(self):
        stage = catalog.LAYOUT
        configs = {catalog.RAW_TRANSFORMER + '/config.json': {'num_layers': 50},
                   stage + '/model_spec.json': {'format_version': 2}, stage + '/linear_branch/config.json': {}}
        for adapter in ('default', 'turbo'):
            relative = stage + '/adapters/' + adapter
            configs[relative + '/adapter_config.json'] = dict(type='lora', version=1, config=dict(
                alpha=1, rank=1, targets=['block'], exact_targets=True))
            write_tensors(self.directory / relative / 'adapter_model.safetensors', [
                (f'block.lora_A.{adapter}.weight', 'BF16', [1, 2]), (f'block.lora_B.{adapter}.weight', 'BF16', [2, 1])])
        for name, config in configs.items():
            write_json(self.directory / name, config)
            write_json(self.root / 'installed' / name, config)
        shard = catalog.RAW_TRANSFORMER + '/model-01.safetensors'
        write_tensors(self.directory / shard, [('block.weight', 'BF16', [2, 2])])
        write_tensors(self.directory / stage / 'linear_branch/model.safetensors', [('branch.weight', 'BF16', [2, 2])])
        write_json(self.directory / catalog.RAW_INDEX, dict(weight_map={'block.weight': 'model-01.safetensors'}))
        return shard

    def _prepared(self):
        names = ['root', *(f'blocks/{n:02}' for n in range(50)), *(f'adaln/{n:02}' for n in range(50))]
        linears = {f'dense.{n}': dict(weight_shape=[2, 2], scale_shape=[1, 2], input_dtype='bfloat16')
                   for n in range(363)}
        groups = []
        for name in names:
            tensors = [(name + '.weight', 'BF16', [1])]
            if name == 'root':
                for linear in linears:
                    tensors.extend([(linear + '.weight_fp8', 'F8_E4M3', [2, 2]),
                                    (linear + '.weight_scale', 'F32', [1, 2])])
            file = name + '.safetensors'
            groups.append(dict(group=name, file=file, linears=linears if name == 'root' else {},
                               **write_tensors(self.directory / file, tensors)))
        manifest = dict(source_id='a' * 64, precision='fp8', scale_granularity='rowwise', groups=groups, linears=linears)
        write_json(self.directory / 'manifest.json', manifest)
        return manifest

    def test_raw_uses_only_indexed_weight_files_and_checks_the_staged_package(self):
        shard = self._raw()
        (self.directory / catalog.RAW_TRANSFORMER / 'unused.safetensors').write_bytes(b'ignore this')
        found = local.inspect(self.directory, self.root, self.machine)
        self.assertTrue(found['compatible'])
        self.assertEqual(found['kind'], 'raw')
        self.assertEqual(found['required_bytes'] - found['copy_bytes'], catalog.RAW_CACHE_RESERVE + catalog.MARGIN)
        targets = {row['target'] for row in found['files']}
        self.assertIn(shard, targets)
        self.assertFalse(any('unused' in target for target in targets))
        stage = self.root / 'copied'
        for row in found['files']:
            target = stage / row['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['source'], target)
        local.validate_import(found, stage, self.machine)
        write_json(stage / catalog.RAW_INDEX, dict(weight_map={'unexpected.weight': 'model-01.safetensors'}))
        with self.assertRaisesRegex(ValueError, 'names disagree with the index'):
            local.validate_import(found, stage, self.machine)

    def test_raw_rejects_missing_or_escaping_transformer_shard(self):
        shard = self._raw()
        (self.directory / shard).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe local model file'):
            local.inspect(self.directory, self.root, self.machine)
        write_json(self.directory / catalog.RAW_INDEX, dict(weight_map={'block.weight': '../outside.safetensors'}))
        with self.assertRaisesRegex(ValueError, 'Invalid VDN-H3 transformer shard name'):
            local.inspect(self.directory, self.root, self.machine)

    def test_prepared_checks_format_and_rejects_a_tensor_with_the_wrong_dtype(self):
        manifest = self._prepared()
        found = local.inspect(self.directory / 'manifest.json', self.root, self.machine)
        self.assertTrue(found['compatible'])
        self.assertEqual(found['format'], 'fp8_rowwise')
        self.assertEqual(found['files'][0]['sha256'], found['manifest_sha256'])
        self.assertEqual(found['files'][1]['sha256'], manifest['groups'][0]['sha256'])
        self.machine['hardware']['capability'] = (12, 0)
        self.assertFalse(local.inspect(self.directory, self.root, self.machine)['compatible'])
        path = self.directory / 'root.safetensors'
        path.write_bytes(path.read_bytes().replace(b'"F8_E4M3"', b'"U8"     ', 1))
        with self.assertRaisesRegex(ValueError, 'Incompatible quantized tensor shape or dtype'):
            local.inspect(self.directory, self.root, self.machine)

    def test_prepared_rejects_unpinned_projection_download(self):
        manifest = self._prepared()
        manifest['adaln_sources'] = {'download': {'repo': 'some/third-party', 'revision': 'a' * 40}}
        write_json(self.directory / 'manifest.json', manifest)
        with patch.object(local, '_official_manifest', return_value=False), self.assertRaisesRegex(ValueError, 'pinned official'):
            local.inspect(self.directory, self.root, self.machine)


if __name__ == '__main__':
    unittest.main()
