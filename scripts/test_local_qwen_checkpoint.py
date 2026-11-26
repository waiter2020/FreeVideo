"""Test logical comparison of Qwen INT8 ConvRot checkpoint headers."""
import unittest

from freevideo_engine.image_gen import qwen_checkpoint as checkpoint


MLP = 'transformer_blocks.1.img_mlp.'
ATTENTION = 'transformer_blocks.1' + checkpoint.ATTENTION_HINT


def projection(name, rows):
    return {
        MLP + name + '.weight': ('I8', (rows, 256)),
        MLP + name + '.weight_scale': ('F32', (rows, 1)),
        MLP + name + '.comfy_quant': ('U8', ()),
    }


class QwenCheckpointTest(unittest.TestCase):
    def test_fused_and_split_headers_have_the_same_logical_projections(self):
        ordinary = {'img_in.weight': ('BF16', (256, 64))}
        fused = {**ordinary, **projection('gate_up', 14), ATTENTION: ('U8', ())}
        split = {**ordinary, **projection('gate_layer', 7), **projection('proj', 7)}
        self.assertEqual(checkpoint.canonical_tensors(fused), split)
        self.assertEqual(checkpoint.canonical_tensors(split), split)
        self.assertIn(ATTENTION, fused)

    def test_mixed_fused_and_split_weights_are_rejected(self):
        for split in ('gate_layer', 'proj'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'Mixed fused and split'):
                checkpoint.canonical_tensors({**projection('gate_up', 14), **projection(split, 7)})

    def test_malformed_fused_layouts_are_rejected(self):
        valid = projection('gate_up', 14)
        for invalid in (
            projection('gate_up', 13),
            {**valid, MLP + 'gate_up.weight': ('I8', (14, 128))},
            {**valid, MLP + 'gate_up.weight': ('BF16', (14, 256))},
            {**valid, MLP + 'gate_up.weight_scale': ('F32', (7, 1))},
            {**valid, MLP + 'gate_up.comfy_quant': ('I8', ())},
            {key: value for key, value in valid.items() if not key.endswith('weight_scale')},
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'fused Qwen gate projection'):
                checkpoint.canonical_tensors(invalid)

    def test_attention_hint_is_optional_but_cannot_be_a_different_tensor_type(self):
        self.assertEqual(checkpoint.canonical_tensors({ATTENTION: ('U8', ())}), {})
        with self.assertRaisesRegex(ValueError, 'Invalid Comfy attention hint tensor'):
            checkpoint.canonical_tensors({ATTENTION: ('F32', (3,))})

    def test_convrot_marker_stays_strict(self):
        marker = checkpoint.CONVROT_METADATA
        self.assertTrue(checkpoint.quant_marker_supported(dict(marker)))
        for invalid in (
            None,
            [],
            {**marker, 'convrot': 1},
            {**marker, 'convrot_groupsize': 256.0},
            {**marker, 'convrot_groupsize': 128},
            {**marker, 'weight_rotated': True},
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(checkpoint.quant_marker_supported(invalid))


if __name__ == '__main__':
    unittest.main()
