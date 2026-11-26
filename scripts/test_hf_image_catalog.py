"""Tests for native image denoiser inspection without model payload downloads."""
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import Request

from freevideo_engine import hf_image_catalog as catalog
from freevideo_engine.image_gen import image_model, image_pack


REPO = 'author/image'
REVISION = 'a' * 40
SHA256 = 'b' * 64


def tensors(entries, metadata=None):
    header = {}
    if metadata is not None:
        header['__metadata__'] = metadata
    end = 0
    for name, dtype, count in entries:
        stop = end + count * catalog.DTYPE_BYTES[dtype]
        header[name] = dict(dtype=dtype, shape=[count], data_offsets=[end, stop])
        end = stop
    data = json.dumps(header).encode()
    return struct.pack('<Q', len(data)) + data + b'\x00' * end


def weight_row(data):
    return dict(size=len(data), lfs=dict(size=len(data), sha256=SHA256))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class ImageCatalogTest(unittest.TestCase):
    def test_gguf_has_a_specific_diagnostic_and_mixed_repositories_can_pass(self):
        model = image_model.QWEN
        stock = tensors([('foo.weight', 'I8', 4)])
        gguf = 'encoder/Qwen-8B-Q8_0.gguf'
        denoiser = 'diffusion_models/full_int8_convrot.safetensors'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            machine = dict(model_root=str(root / 'models'))
            builtin = next(row['file'] for row in image_pack.rows(group='image') if row['component'] == 'dit')
            write(root / 'models' / 'image_gen' / builtin, stock)
            info = dict(id=REPO, sha=REVISION)
            entries = {gguf: {'size': 10}}
            with patch.object(catalog.image_pack, 'ready', return_value=True), patch.object(
                    catalog.hub, '_info', return_value=(info, entries)), patch.object(
                    catalog, '_range', side_effect=lambda row, start, end, token: stock[start:end + 1]) as fetch:
                for ref in (REPO, f'https://huggingface.co/{REPO}/blob/main/{gguf}'):
                    with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, 'contains GGUF'):
                        catalog.inspect(ref, root, machine, model)
                with self.assertRaisesRegex(ValueError, 'No compatible denoiser'):
                    catalog.inspect(REPO, root, machine, model, subfolder='diffusion_models')
                fetch.assert_not_called()
                entries[denoiser] = weight_row(stock)
                result = catalog.inspect(REPO, root, machine, model)
                self.assertEqual(result['path'], denoiser)
                self.assertTrue(result['compatible'])

    def test_qwen_repo_and_exact_file_report_attention_metadata_difference(self):
        model = image_model.QWEN
        stock = tensors([('foo.weight', 'I8', 4), ('foo' + catalog.ATTENTION_HINT, 'U8', 11)])
        remote = tensors([('foo.weight', 'I8', 4)])
        source = 'qwen-image-2.1-INT8-ConvRot-comfy.safetensors'
        unrelated = [f'q{i}.safetensors' for i in range(catalog.MAX_CHOICES)]
        entries = {name: weight_row(remote) for name in [*unrelated, source]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            machine = dict(model_root=str(root / 'models'))
            builtin = next(row['file'] for row in image_pack.rows(group='image') if row['component'] == 'dit')
            write(root / 'models' / 'image_gen' / builtin, stock)
            info = dict(id=REPO, sha=REVISION)
            self.assertEqual(catalog._choices(entries, '', builtin, model)[0], source)
            with patch.object(catalog.image_pack, 'ready', return_value=True), patch.object(
                    catalog.hub, '_info', return_value=(info, entries)), patch.object(
                    catalog, '_range', side_effect=lambda row, start, end, token: remote[start:end + 1]):
                for reference in (REPO, f'https://huggingface.co/{REPO}/blob/main/{source}'):
                    with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, 'attention metadata differs'):
                        catalog.inspect(reference, root, machine, model)

    def test_qwen_normalizes_native_keys_and_marker_length(self):
        model = image_model.QWEN
        source = 'diffusion_models/derivative_int8_convrot.safetensors'
        left = tensors([('foo.weight', 'I8', 4), ('foo.comfy_quant', 'U8', 11)])
        right = tensors([('model.diffusion_model.foo.weight', 'I8', 4),
                         ('model.diffusion_model.foo.comfy_quant', 'U8', 12)])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            machine = dict(model_root=str(root / 'models'))
            builtin = next(row['file'] for row in image_pack.rows(group='image') if row['component'] == 'dit')
            write(root / 'models' / 'image_gen' / builtin, left)
            info = dict(id=REPO, sha=REVISION, cardData={'license': 'apache-2.0'})
            with patch.object(catalog.image_pack, 'ready', return_value=True), patch.object(
                    catalog.hub, '_info', return_value=(info, {source: weight_row(right)})), patch.object(
                    catalog, '_range', side_effect=lambda row, start, end, token: right[start:end + 1]):
                result = catalog.inspect(f'https://huggingface.co/{REPO}/blob/main/{source}', root, machine, model)
                self.assertEqual(result['repo'], REPO)
                self.assertEqual(result['revision'], REVISION)
                self.assertEqual(result['weights'], 'image_gen/' + builtin)
                self.assertEqual(result['files'][0]['file'], source)
                self.assertEqual(result['files'][0]['target'], result['weights'])
                self.assertTrue(all(row['file'] != builtin for row in result['base_files']))
                pack = root / 'profile' / 'image_gen'
                write(pack / builtin, right)
                catalog.validate_import(result, pack, machine)
                write(pack / builtin, tensors([('foo.weight', 'U8', 4)]))
                with self.assertRaisesRegex(ValueError, 'staged image checkpoint'):
                    catalog.validate_import(result, pack, machine)

    def test_moss_accepts_resharding_without_remote_configuration_or_code(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            machine = dict(model_root=str(root / 'models'))
            model = image_model.MOSS
            stock = root / 'models' / 'image_gen' / catalog.MOSS_DIRECTORY
            write(stock / catalog.SINGLE, tensors([('a', 'BF16', 4), ('b', 'BF16', 8)]))
            two = {'model-a.safetensors': tensors([('a', 'BF16', 4)]),
                   'model-b.safetensors': tensors([('b', 'BF16', 8)])}
            index = dict(weight_map={'a': 'model-a.safetensors', 'b': 'model-b.safetensors'})
            body = json.dumps(index).encode()
            prefix = 'transformer/'
            entries = {prefix + catalog.INDEX: dict(size=len(body), blobId=REVISION),
                       **{prefix + k: weight_row(v) for k, v in two.items()},
                       'pipeline_moss_image.py': dict(size=100, blobId=REVISION)}
            info = dict(id=REPO, sha=REVISION)
            def fetch(row, start, end, token):
                return two[Path(row['file']).name][start:end + 1]
            with patch.object(catalog.image_pack, 'ready', return_value=True), patch.object(
                    catalog.hub, '_info', return_value=(info, entries)), patch.object(
                    catalog.hub, '_read', return_value=(index, SHA256)), patch.object(catalog, '_range', side_effect=fetch):
                result = catalog.inspect(REPO, root, machine, model)
                self.assertEqual(result['weights'], 'image_gen/' + catalog.MOSS_DIRECTORY)
                self.assertEqual({row['file'] for row in result['files']}, {prefix + catalog.INDEX,
                                  *(prefix + name for name in two)})
                self.assertIn('moss-image/pipeline_moss_image.py', [row['file'] for row in result['base_files']])
                profile = root / 'profile' / 'image_gen'
                write(profile / catalog.MOSS_DIRECTORY / catalog.INDEX, body)
                for name, data in two.items():
                    write(profile / catalog.MOSS_DIRECTORY / name, data)
                catalog.validate_import(result, profile, machine)

    def test_moss_index_rejects_paths_outside_the_transformer(self):
        for value in ('../other.safetensors', '/other.safetensors', 'folder/other.safetensors',
                      'C:\\weights.safetensors'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                catalog._weight_map({'weight_map': {'weight': value}})
        with self.assertRaisesRegex(ValueError, 'distinct on all platforms'):
            catalog._weight_map({'weight_map': {'a': 'weights.safetensors', 'b': 'Weights.safetensors'}})

    def test_weight_redirect_keeps_range_and_discards_authorization(self):
        request = Request('https://huggingface.co/author/image/resolve/main/model.safetensors',
                          headers={'Range': 'bytes=0-7'})
        request.add_unredirected_header('Authorization', 'Bearer hf_example')
        redirect = catalog._WeightRedirect()
        redirected = redirect.redirect_request(request, None, 302, '', {}, 'https://cas-bridge.xethub.hf.co/file')
        self.assertEqual(redirected.get_header('Range'), 'bytes=0-7')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaises(ValueError):
            redirect.redirect_request(request, None, 302, '', {}, 'https://other.example/file')

    def test_h3_uses_the_existing_cuda_only_family_support(self):
        with self.assertRaisesRegex(ValueError, 'Apple Silicon'):
            catalog.inspect(REPO, '.', {'device_backend': 'mps'}, image_model.H3)


if __name__ == '__main__':
    unittest.main()
