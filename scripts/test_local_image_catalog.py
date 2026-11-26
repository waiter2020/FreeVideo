"""Local image inspection uses model metadata and only the chosen files."""
import hashlib
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from freevideo_engine import local_image_catalog as catalog
from freevideo_engine.hf_image_catalog import INDEX, MOSS_DIRECTORY, SINGLE, _selection, validate_import
from freevideo_engine.image_gen import image_model
from freevideo_engine.image_gen.qwen_checkpoint import ATTENTION_HINT, COMFY_ATTENTION_HINT, CONVROT_METADATA
from scripts.test_hf_image_catalog import tensors, write


def gate_checkpoint(*, fused=False, marker=CONVROT_METADATA):
    prefix = 'transformer_blocks.0.img_mlp.'
    header, payload = {}, bytearray()

    def add(key, dtype, shape, data):
        start = len(payload)
        payload.extend(data)
        header[key] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])

    names = [('gate_up', 4)] if fused else [('gate_layer', 2), ('proj', 2)]
    for name, rows in names:
        add(prefix + name + '.weight', 'I8', [rows, 256], bytes(rows * 256))
        add(prefix + name + '.weight_scale', 'F32', [rows, 1], bytes(rows * 4))
        raw = json.dumps(marker).encode()
        add(prefix + name + '.comfy_quant', 'U8', [len(raw)], raw)
    if fused:
        raw = json.dumps(COMFY_ATTENTION_HINT).encode()
        add('transformer_blocks.0' + ATTENTION_HINT, 'U8', [len(raw)], raw)
    data = json.dumps(header).encode()
    return struct.pack('<Q', len(data)) + data + payload


class LocalImageCatalogTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.machine = {'model_root': str(self.root / 'models')}
        self.inputs = self.root / '本地权重'
        ready = patch('freevideo_engine.image_gen.image_pack.ready', return_value=True)
        ready.start()
        self.addCleanup(ready.stop)

    def stock(self, model, entries):
        pack, name, _ = _selection(model, self.machine)
        target = pack / name
        write(target / SINGLE if model == image_model.MOSS else target, tensors(entries))
        return 'image_gen/' + name

    def test_single_files_use_native_destination_and_keep_installed_companions(self):
        for model in (image_model.H3, image_model.QWEN, image_model.MOSS):
            with self.subTest(model=model):
                target = self.stock(model, [('tensor', 'BF16', 4)])
                local = self.inputs / (model + ' 重命名.SAFETENSORS')
                write(local, tensors([('tensor', 'BF16', 4)]))
                result = catalog.inspect(local, self.root, self.machine, model)
                self.assertTrue(result['compatible'])
                self.assertEqual(result['source_kind'], 'local')
                self.assertEqual(result['label'], local.name)
                self.assertEqual(result['weights'], target)
                file, = result['files']
                self.assertEqual(file['target'], target + '/' + SINGLE if model == image_model.MOSS else target)
                self.assertEqual(Path(file['source']).resolve(), local.resolve())
                self.assertEqual(file['bytes'], result['copy_bytes'])
                self.assertGreater(result['required_bytes'], result['copy_bytes'])
                self.assertTrue(result['base_files'])

    def test_moss_index_copies_only_referenced_shards_and_normalizes_the_index_name(self):
        expected = self.stock(image_model.MOSS, [('one', 'BF16', 4), ('two', 'BF16', 8)])
        index = self.inputs / '替代分片.safetensors.index.json'
        data = json.dumps({'weight_map': {'one': 'first.safetensors', 'two': 'second.safetensors'}}).encode()
        write(index, data)
        write(index.parent / 'first.safetensors', tensors([('one', 'BF16', 4)]))
        write(index.parent / 'second.safetensors', tensors([('two', 'BF16', 8)]))
        write(index.parent / 'ignored.safetensors', b'ignored')
        write(index.parent / 'pipeline_moss_image.py', b'ignored')
        result = catalog.inspect(index, self.root, self.machine, image_model.MOSS)
        self.assertEqual(result['weights'], expected)
        self.assertEqual(result['files'][0]['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual({row['target'] for row in result['files']},
                         {expected + '/' + name for name in (INDEX, 'first.safetensors', 'second.safetensors')})
        staging = self.root / 'profile'
        for row in result['files']:
            write(staging / row['target'], Path(row['source']).read_bytes())
        validate_import(result, staging / 'image_gen', self.machine)
        self.assertEqual((staging / 'image_gen' / MOSS_DIRECTORY / INDEX).read_bytes(), data)

    def test_moss_reports_missing_shards_and_rejects_escape_paths(self):
        self.stock(image_model.MOSS, [('one', 'BF16', 4)])
        index = self.inputs / INDEX
        write(index, json.dumps({'weight_map': {'one': 'missing.safetensors'}}).encode())
        with self.assertRaisesRegex(ValueError, 'Missing MOSS-Image shard: missing'):
            catalog.inspect(index, self.root, self.machine, image_model.MOSS)
        write(index, json.dumps({'weight_map': {'one': '../other.safetensors'}}).encode())
        with self.assertRaisesRegex(ValueError, 'Invalid .*(path|folder)'):
            catalog.inspect(index, self.root, self.machine, image_model.MOSS)

    def test_qwen_split_gate_weights_pass_inspection_and_staged_validation(self):
        model = image_model.QWEN
        pack, name, _ = _selection(model, self.machine)
        write(pack / name, gate_checkpoint(fused=True))
        local = self.inputs / 'split.safetensors'
        write(local, gate_checkpoint())
        result = catalog.inspect(local, self.root, self.machine, model)
        self.assertTrue(result['compatible'])
        staged = self.root / 'profile' / 'image_gen'
        write(staged / name, local.read_bytes())
        catalog.validate_import(result, staged, self.machine)
        invalid = gate_checkpoint(marker=dict(CONVROT_METADATA, convrot=False))
        write(local, invalid)
        write(staged / name, invalid)
        with self.assertRaisesRegex(ValueError, 'Unsupported Qwen image checkpoint quantization'):
            catalog.inspect(local, self.root, self.machine, model)
        with self.assertRaisesRegex(ValueError, 'Unsupported Qwen image checkpoint quantization'):
            catalog.validate_import(result, staged, self.machine)

    def test_gguf_lora_encoder_wrong_shape_and_unexpected_index_are_rejected(self):
        model = image_model.QWEN
        self.stock(model, [('native.weight', 'I8', 4)])
        cases = (
            ('qwen.gguf', b'GGUF', 'contains GGUF weights'),
            ('lora.safetensors', tensors([('base.block.lora_A.weight', 'BF16', 4)]), 'LoRA adapter'),
            ('encoder.safetensors', tensors([('model.layers.1.weight', 'F32', 4)]), 'text encoder weights'),
            ('wrong.safetensors', tensors([('native.weight', 'I8', 8)]), 'tensor names, shapes or storage types'),
            ('wrong.safetensors.index.json', b'{}', 'Only MOSS-Image accepts a shard index'),
        )
        for name, data, error in cases:
            with self.subTest(name=name):
                local = self.inputs / name
                write(local, data)
                with self.assertRaisesRegex(ValueError, error):
                    catalog.inspect(local, self.root, self.machine, model)


if __name__ == '__main__':
    unittest.main()
