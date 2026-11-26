"""Metadata shared by the Qwen catalog and native INT8 ConvRot loader."""

ATTENTION_HINT = '.attn.comfy_attention.config'
CONVROT_METADATA = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}
COMFY_ATTENTION_HINT = {'attention': 'comfy_kitchen_int8'}


def quant_marker_supported(marker):
    return (isinstance(marker, dict) and marker == CONVROT_METADATA
            and marker.get('convrot') is True and type(marker.get('convrot_groupsize')) is int)


def canonical_tensors(tensors):
    """Compare the loader's logical projections; marker lengths have already been validated."""
    canonical, fused, projections = {}, {}, set()
    for key, signature in tensors.items():
        if key.endswith(ATTENTION_HINT):
            if signature != ('U8', ()):
                raise ValueError('Invalid Comfy attention hint tensor: ' + key)
            continue
        projection, _, part = key.rpartition('.')
        projections.add(projection)
        if projection.endswith('.img_mlp.gate_up'):
            fused.setdefault(projection, {})[part] = signature
        else:
            canonical[key] = signature

    for projection, parts in fused.items():
        prefix = projection.removesuffix('gate_up')
        targets = (prefix + 'gate_layer', prefix + 'proj')
        if any(target in projections for target in targets):
            raise ValueError('Mixed fused and split Qwen gate projections: ' + projection)
        if parts.keys() != {'weight', 'weight_scale', 'comfy_quant'}:
            raise ValueError('Incomplete fused Qwen gate projection: ' + projection)
        dtype, weight = parts['weight']
        if dtype != 'I8' or len(weight) != 2:
            raise ValueError('Invalid fused Qwen gate projection weight: ' + projection)
        rows, width = weight
        if (rows <= 0 or rows % 2 or width <= 0 or width % CONVROT_METADATA['convrot_groupsize']
                or parts['weight_scale'] != ('F32', (rows, 1)) or parts['comfy_quant'] != ('U8', ())):
            raise ValueError('Invalid fused Qwen gate projection shape or marker: ' + projection)
        for target in targets:
            canonical[target + '.weight'] = ('I8', (rows // 2, width))
            canonical[target + '.weight_scale'] = ('F32', (rows // 2, 1))
            canonical[target + '.comfy_quant'] = parts['comfy_quant']
    return canonical
