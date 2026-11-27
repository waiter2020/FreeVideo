"""Picture model families, package identity and supported canvas sizes."""
import json
import math
from pathlib import Path

from ..monitoring import save

QWEN = 'qwen-image-2.1-turbo'
H3 = 'minimax-h3-image'
MOSS = 'moss-image-sft'
MODELS = (H3, QWEN, MOSS)
FILE = 'image-model.json'

# Side of the square whose area each size tier asks for. H3 Image starts at 4 MP (2048 x 2048) and goes to 16 MP;
# the same table as the picture page. Qwen keeps its own sizes.
H3_AREAS = {'standard': 2048, 'high': 4096}
# MOSS-Image: side of the square whose area each tier asks for (1 MP, 4 MP), in steps of 16.
MOSS_AREAS = {'standard': 1024, 'high': 2048}
RATIOS = {'1:1': (1, 1), '16:9': (16, 9), '9:16': (9, 16), '4:3': (4, 3), '3:4': (3, 4)}


def remembered(root):
    try:
        value = json.loads((Path(root) / FILE).read_text(encoding='utf-8')).get('model')
    except (OSError, ValueError, AttributeError):
        return None
    return value if value in MODELS else None


def remember(root, model):
    """Keep the model the page last generated with; a missing or unknown id changes nothing."""
    if model in MODELS and model != remembered(root):
        try:
            save(Path(root) / FILE, dict(model=model))
        except OSError:
            pass  # The choice only goes unremembered.


def open_package(machine, model, package=None, expected=None):
    """Resolve picture weights and check the identity stored when the request was accepted."""
    from ..image_components import ComponentError, digest, resolve
    from ..model_packages import PackageError, resolve_image

    try:
        if model is None:
            resolved = resolve_image(machine, package_id=package)
        else:
            resolved = resolve(machine, model, package=package) if package else resolve(machine, model)
    except PackageError as error:
        raise ComponentError(str(error)) from error
    identity = digest(resolved['identity'])
    if expected is not None and identity != expected:
        raise ComponentError('The image model package changed after this request was accepted. Try again.')
    return resolved, identity


def choose(info, prefer=None):
    """Choose the caller's preferred family, the active package's family, or a ready default."""
    info = info or {}
    ready = [row['id'] for row in info.get('models') or ()
             if row.get('ready') and row.get('available', True) and 'component_error' not in row and row.get('id') in MODELS]
    active = next((row for row in info.get('image_packages') or () if row.get('active')), None)
    if active and not active.get('ready'):
        ready = [family for family in ready if family != active.get('family')]
    if not ready:
        return QWEN
    if prefer in ready:
        return prefer
    if active and active.get('family') in ready:
        return active['family']
    return H3 if H3 in ready else ready[0]


def size(model, aspect, tier='standard'):
    """Width and height for an aspect at a tier: H3 Image at 4 or 16 MP in steps of 32, else None (Qwen's table)."""
    if model == MOSS:
        ratio_w, ratio_h = RATIOS[aspect]
        side = MOSS_AREAS.get(tier, MOSS_AREAS['standard'])
        scale = math.sqrt(side * side / (ratio_w * ratio_h))
        return tuple(max(256, min(2048, round(value * scale / 16) * 16)) for value in (ratio_w, ratio_h))
    if model != H3:
        return None
    ratio_w, ratio_h = RATIOS[aspect]
    side = H3_AREAS.get(tier, H3_AREAS['standard'])
    scale = math.sqrt(side * side / (ratio_w * ratio_h))
    return tuple(max(32, round(value * scale / 32) * 32) for value in (ratio_w, ratio_h))


def at_least_four_megapixels(model, width, height):
    """H3 Image is asked for the same shape at 4 MP when the request is smaller; the picture is not scaled back."""
    if model != H3 or width * height >= H3_AREAS['standard'] ** 2:
        return width, height
    scale = math.sqrt(H3_AREAS['standard'] ** 2 / (width * height))
    return tuple(max(32, round(value * scale / 32) * 32) for value in (width, height))
