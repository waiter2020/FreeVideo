"""Image component inspection uses native signatures and checksum-pinned inputs."""
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from freevideo_engine import image_component_catalog as catalog
from freevideo_engine.image_gen import image_model
from freevideo_engine.image_gen.qwen_checkpoint import CONVROT_METADATA
from scripts.test_hf_image_catalog import REPO, REVISION, tensors, weight_row, write


def quantized(prefix, marker, vision=True):
    raw = json.dumps(marker).encode()
    end = len(raw)
    header = {prefix + '.comfy_quant': dict(dtype='U8', shape=[end], data_offsets=[0, end])}
    if vision:
        header['model.visual.patch.weight'] = dict(dtype='BF16', shape=[1], data_offsets=[end, end + 2])
        raw += bytes(2)
    data = json.dumps(header).encode()
    return struct.pack('<Q', len(data)) + data + raw


def h3_nvfp4(awq=False, vision=True, global_scale=None, marker=None):
    prefix = 'model.layers.0.self_attn.o_proj'
    if global_scale is None:
        global_scale = [] if awq else [1]
    if marker is None:
        marker = {'format': 'nvfp4', **({'full_precision_matrix_mult': True} if awq else {})}
    marker = json.dumps(marker).encode()
    embed = b'{"format":"int8_tensorwise"}'
    tensors = {
        prefix + '.weight': ('U8', [2, 4], bytes(8)),
        prefix + '.weight_scale': ('F8_E4M3', [2, 1], bytes(2)),
        prefix + '.weight_scale_2': ('F32', global_scale, bytes(4 * (global_scale[0] if global_scale else 1))),
        prefix + '.comfy_quant': ('U8', [len(marker)], marker),
        'model.embed_tokens.weight': ('I8', [3, 2], bytes(6)),
        'model.embed_tokens.weight_scale': ('F32', [3, 1] if awq else [], bytes(12 if awq else 4)),
        'model.embed_tokens.comfy_quant': ('U8', [len(embed)], embed),
    }
    if awq:
        tensors[prefix + '.pre_quant_scale'] = ('BF16', [8], bytes(16))
    if vision:
        tensors['visual.patch_embed.proj.weight'] = ('BF16', [2, 1], bytes(4))
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    data = json.dumps(header).encode()
    return struct.pack('<Q', len(data)) + data + payload


class ComponentCatalogTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.machine = dict(model_root=self.root / 'models' / 'vdn')

    def test_qwen_encoder_normalizes_language_prefix_and_verifies_lazy_vision_and_markers(self):
        model, role = image_model.QWEN, catalog.TEXT_ENCODER
        stock = catalog.stock_paths(self.machine, model, role)
        write(stock, quantized('model.layers.0.q_proj', CONVROT_METADATA))
        local = self.root / 'input' / 'new-encoder.safetensors'
        content = quantized('model.language_model.layers.0.q_proj', CONVROT_METADATA)
        write(local, content)
        result = catalog.inspect_local(local, self.root, self.machine, model, role)
        self.assertEqual(result['weights'], stock.name)
        self.assertEqual(result['component'], role)
        self.assertEqual(result['files'][0]['target'], stock.name)
        self.assertEqual(result['base_files'], [])
        staged = self.root / 'stage'
        write(staged / stock.name, content)
        catalog.validate_import(result, staged, self.machine)
        write(local, quantized('model.layers.0.q_proj', CONVROT_METADATA, vision=False))
        with self.assertRaisesRegex(ValueError, 'tensor names'):
            catalog.inspect_local(local, self.root, self.machine, model, role)
        write(staged / stock.name, quantized('model.layers.0.q_proj', dict(CONVROT_METADATA, convrot=False)))
        with self.assertRaisesRegex(ValueError, 'quantization markers'):
            catalog.validate_import(result, staged, self.machine)

    def test_h3_uses_shared_encoder_and_fp8_storage_signatures(self):
        model, role = image_model.H3, catalog.TEXT_ENCODER
        self.machine.update(encoder_model_root=self.root / 'isolated-encoder', encoder='pinned-encoder.safetensors')
        stock = catalog.stock_paths(self.machine, model, role)
        self.assertEqual(stock, self.root / 'isolated-encoder' / 'text_encoders' / 'pinned-encoder.safetensors')
        write(stock, tensors([('layer.weight_scale', 'F8_E4M3', 4)]))
        local = self.root / 'new-h3.safetensors'
        write(local, stock.read_bytes())
        self.assertEqual(catalog.inspect_local(local, self.root, self.machine, model, role)['format'], 'nvfp4_awq')
        write(local, tensors([('layer.weight_scale', 'F8_E5M2', 4)]))
        with self.assertRaisesRegex(ValueError, 'storage types'):
            catalog.inspect_local(local, self.root, self.machine, model, role)

    def test_h3_identifies_plain_nvfp4_and_preserves_awq_validation(self):
        model, role = image_model.H3, catalog.TEXT_ENCODER
        stock = catalog.stock_paths(self.machine, model, role)
        awq = h3_nvfp4(awq=True)
        write(stock, awq)
        local, staged = self.root / 'heretic.safetensors', self.root / 'stage'
        plain = h3_nvfp4()
        write(local, plain)
        candidate = catalog.inspect_local(local, self.root, self.machine, model, role)
        self.assertEqual(candidate['format'], 'nvfp4')
        self.assertEqual(candidate['weights'], stock.name)
        write(staged / stock.name, plain)
        catalog.validate_import(candidate, staged, self.machine)
        write(staged / stock.name, awq)
        with self.assertRaisesRegex(ValueError, 'format changed'):
            catalog.validate_import(candidate, staged, self.machine)
        write(local, awq)
        self.assertEqual(catalog.inspect_local(local, self.root, self.machine, model, role)['format'], 'nvfp4_awq')

        for content, error in ((h3_nvfp4(vision=False), 'tensor names'),
                               (h3_nvfp4(global_scale=[2]), 'storage types'),
                               (h3_nvfp4(marker={'format': 'int8_tensorwise'}), 'quantization markers'),
                               (h3_nvfp4(awq=True, marker={'format': 'nvfp4'}), 'quantization markers')):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                write(local, content)
                catalog.inspect_local(local, self.root, self.machine, model, role)

    def test_h3_plain_nvfp4_remote_layout_is_rechecked_locally(self):
        model, role = image_model.H3, catalog.TEXT_ENCODER
        stock = catalog.stock_paths(self.machine, model, role)
        write(stock, h3_nvfp4(awq=True))
        plain = h3_nvfp4(marker={'format': 'nvfp4', 'full_precision_matrix_mult': False})
        name = 'text_encoders/community.safetensors'
        with (patch.object(catalog.hub, '_info', return_value=(dict(id=REPO, sha=REVISION), {name: weight_row(plain)})),
              patch.object(catalog.native, '_range', side_effect=lambda row, start, end, token: plain[start:end + 1])):
            remote = catalog.inspect(f'https://huggingface.co/{REPO}/blob/main/{name}', self.root, self.machine, model, role)
        self.assertEqual(remote['format'], 'nvfp4')
        self.assertEqual(remote['revision'], REVISION)
        staged = self.root / 'stage'
        write(staged / stock.name, plain)
        catalog.validate_import(remote, staged, self.machine)
        write(staged / stock.name, h3_nvfp4(marker={'format': 'nvfp4', 'full_precision_matrix_mult': 'false'}))
        with self.assertRaisesRegex(ValueError, 'quantization markers'):
            catalog.validate_import(remote, staged, self.machine)

    def test_moss_inspects_index_shards_and_pins_hub_files(self):
        model, role = image_model.MOSS, catalog.TEXT_ENCODER
        stock = catalog.stock_paths(self.machine, model, role)
        single = catalog.MOSS_FILES[role]
        write(stock / single, tensors([('one', 'BF16', 4), ('two', 'BF16', 8)]))
        directory = self.root / 'inputs'
        weights = {'a.safetensors': tensors([('one', 'BF16', 4)]), 'b.safetensors': tensors([('two', 'BF16', 8)])}
        index = {'weight_map': {'one': 'a.safetensors', 'two': 'b.safetensors'}}
        index_name = 'custom.safetensors.index.json'
        write(directory / index_name, json.dumps(index).encode())
        for name, content in weights.items():
            write(directory / name, content)
        local = catalog.inspect_local(directory / index_name, self.root, self.machine, model, role)
        self.assertEqual(local['weights'], '.')
        self.assertEqual({row['target'] for row in local['files']}, {single + '.index.json', *weights})
        staged = self.root / 'stage'
        for row in local['files']:
            write(staged / row['target'], Path(row['source']).read_bytes())
        catalog.validate_import(local, staged, self.machine)
        write(staged / 'a.safetensors', tensors([('wrong', 'BF16', 4)]))
        with self.assertRaisesRegex(ValueError, 'disagree with the index'):
            catalog.validate_import(local, staged, self.machine)

        entries = {'text_encoder/' + name: weight_row(content) for name, content in weights.items()}
        entries['text_encoder/' + index_name] = dict(size=20, blobId='b' * 40)

        def read_range(row, start, end, _token):
            return weights[Path(row['file']).name][start:end + 1]

        with (patch.object(catalog.hub, '_info', return_value=(dict(id=REPO, sha=REVISION), entries)),
              patch.object(catalog.hub, '_read', return_value=(index, 'c' * 64)),
              patch.object(catalog.native, '_range', side_effect=read_range)):
            url = f'https://huggingface.co/{REPO}/blob/main/text_encoder/{index_name}'
            remote = catalog.inspect(url, self.root, self.machine, model, role)
            from_repo = catalog.inspect(f'https://huggingface.co/{REPO}', self.root, self.machine, model, role)
        self.assertEqual({row['revision'] for row in remote['files']}, {REVISION})
        self.assertEqual(remote['files'][0]['sha256'], 'c' * 64)
        self.assertEqual({row['target'] for row in remote['files']}, {single + '.index.json', *weights})
        self.assertEqual(from_repo['files'], remote['files'])

    def test_qwen_vae_rejects_other_dtype_or_configuration(self):
        model, role = image_model.QWEN, catalog.VAE
        stock = catalog.stock_paths(self.machine, model, role)
        write(stock, tensors([('decoder.weight', 'BF16', 4)]))
        local = self.root / 'vae.safetensors'
        write(local, stock.read_bytes())
        self.assertEqual(catalog.inspect_local(local, self.root, self.machine, model, role)['format'], 'bf16')
        for invalid in (tensors([('decoder.weight', 'F16', 4)]),
                        tensors([('decoder.weight', 'BF16', 4)], {'config': '{"z_dim": 16}'})):
            write(local, invalid)
            with self.assertRaisesRegex(ValueError, 'built-in VAE'):
                catalog.inspect_local(local, self.root, self.machine, model, role)

    def test_moss_vae_requires_a_single_file(self):
        model, role = image_model.MOSS, catalog.VAE
        stock = catalog.stock_paths(self.machine, model, role)
        write(stock / catalog.MOSS_FILES[role], tensors([('decoder.weight', 'F32', 4)]))
        local = self.root / 'weights.safetensors.index.json'
        write(local, b'{}')
        with self.assertRaisesRegex(ValueError, 'single safetensors'):
            catalog.inspect_local(local, self.root, self.machine, model, role)


if __name__ == '__main__':
    unittest.main()
