"""Inspect compatible Hugging Face denoisers for the three native image models."""
import hashlib
import json
import math
from pathlib import Path
import struct
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import hf_auth, hf_video_catalog as hub
from .image_gen import image_model, image_pack
from .image_gen.qwen_checkpoint import ATTENTION_HINT, COMFY_ATTENTION_HINT, quant_marker_supported
from .long_paths import io_path


GROUPS = {image_model.QWEN: 'image', image_model.H3: 'h3-image:int8', image_model.MOSS: 'moss-image'}
FORMATS = {image_model.QWEN: 'int8_convrot', image_model.H3: 'int8_convrot', image_model.MOSS: 'bf16'}
INDEX = 'diffusion_pytorch_model.safetensors.index.json'
SINGLE = 'diffusion_pytorch_model.safetensors'
MOSS_DIRECTORY = 'moss-image/transformer'
MAX_HEADER = 4 * 1024 * 1024
MAX_SHARDS = 16
MAX_CHOICES = 6
DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2, 'I32': 4,
               'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1}


def search(query, *, token=None):
    return hub.search(query, token=token)


def _reference(reference, revision, subfolder):
    # A model file copied from the Hub UI can be more precise than a repository link.
    if isinstance(reference, str) and reference.strip().startswith('https://huggingface.co/'):
        parts = urlsplit(reference.strip())
        route = parts.path.split('/')
        if len(route) > 5 and route[3] in ('blob', 'resolve'):
            route[3] = 'tree'
            reference = parts._replace(path='/'.join(route)).geturl()
    return hub.parse_reference(reference, revision=revision, subfolder=subfolder)


class _WeightRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        url = urlsplit(newurl)
        host = (url.hostname or '').lower()
        if (url.scheme != 'https' or url.username is not None or url.password is not None
                or url.port not in (None, 443) or not (host == 'huggingface.co' or host.endswith('.huggingface.co')
                                                      or host.endswith('.hf.co'))):
            raise ValueError('Hugging Face sent the model file to an unsupported download host.')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _range(row, start, end, token):
    length = end - start + 1
    request = Request(hub.HUB + '/' + row['repo'] + '/resolve/' + row['revision'] + '/' + quote(row['file'], safe='/'),
                      headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'FreeVideo/ImageModelCatalog'})
    if token:
        # Supplemental headers do not follow a redirect to a signed CDN URL.
        request.add_unredirected_header('Authorization', 'Bearer ' + token)
    try:
        with build_opener(_WeightRedirect()).open(request, timeout=10) as response:
            expected = f'bytes {start}-{end}/{row["bytes"]}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Hugging Face did not return the requested weight header range.')
            data = response.read(length + 1)
    except (HTTPError, URLError, OSError) as error:
        raise ValueError('Could not inspect the image tensor header on Hugging Face.') from error
    if len(data) != length:
        raise ValueError('Hugging Face returned an incomplete image tensor header.')
    return data


def _header_size(data, total):
    size = struct.unpack('<Q', data)[0] if len(data) == 8 else 0
    if not 2 <= size <= min(MAX_HEADER, total - 8):
        raise ValueError('Invalid image safetensors header length.')
    return size


def _header(data, size, total, model, *, read_marker=None, markers=None, parse_config=False):
    value = hub._json(data)
    if len(data) != size or not isinstance(value, dict):
        raise ValueError('Invalid image safetensors header.')
    metadata = value.pop('__metadata__', {})
    if not isinstance(metadata, dict):
        raise ValueError('Invalid image safetensors metadata.')
    config = {}
    if (model == image_model.QWEN or parse_config) and 'config' in metadata:
        raw = metadata['config']
        config = hub._json(raw) if isinstance(raw, str) else None
        if not isinstance(config, dict):
            raise ValueError('Invalid Qwen image checkpoint configuration.')
        config = {key: entry for key, entry in config.items() if not key.startswith('_')}
    shapes, offsets = {}, []
    for name, row in value.items():
        dtype = row.get('dtype') if isinstance(row, dict) else None
        shape = row.get('shape') if isinstance(row, dict) else None
        span = row.get('data_offsets') if isinstance(row, dict) else None
        if (not isinstance(dtype, str) or dtype not in DTYPE_BYTES or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape) or len(shape) > 8
                or not isinstance(span, list) or len(span) != 2
                or any(type(n) is not int or n < 0 for n in span)
                or span[1] > total - size - 8
                or span[1] - span[0] != math.prod(shape) * DTYPE_BYTES[dtype]):
            raise ValueError('Invalid image safetensors tensor layout.')
        offsets.append(tuple(span))
        key = name.removeprefix('model.diffusion_model.') if model == image_model.QWEN else name
        if not key or key in shapes:
            raise ValueError('Duplicate or empty image tensor name.')
        quant_marker = key.endswith('.comfy_quant')
        if quant_marker or key.endswith(ATTENTION_HINT):
            marker_vector = dtype == 'U8' and len(shape) == 1
            if read_marker is not None:
                if not marker_vector or not 0 < shape[0] <= 4096:
                    raise ValueError('Invalid Qwen image checkpoint marker tensor.')
                marker = hub._json(read_marker(*span))
                if markers is not None:
                    markers[key] = marker
                elif not (quant_marker_supported(marker) if quant_marker else marker == COMFY_ATTENTION_HINT):
                    raise ValueError('Unsupported Qwen image checkpoint quantization or attention metadata.')
            if marker_vector and not 0 < shape[0] <= 4096:
                raise ValueError('Invalid image quantization marker length.')
            if marker_vector:
                shape = []  # JSON whitespace may vary without changing the marker.
        shapes[key] = dtype, tuple(shape)
    end = 0
    for start, stop in sorted(offsets):
        if start != end:
            raise ValueError('Image safetensors offsets are not contiguous.')
        end = stop
    if not shapes or end != total - size - 8:
        raise ValueError('Invalid image safetensors payload length.')
    return shapes, config


def _local_header(path, model, *, verify_qwen_markers=False, markers=None, parse_config=False):
    file = io_path(path)
    with file.open('rb') as stream:
        total = file.stat().st_size
        size = _header_size(stream.read(8), total)

        def read_marker(start, end):
            stream.seek(8 + size + start)
            return stream.read(end - start)

        reader = read_marker if verify_qwen_markers or markers is not None else None
        return _header(stream.read(size), size, total, model, read_marker=reader, markers=markers,
                       parse_config=parse_config)


def _remote_header(row, model, token, *, parse_config=False):
    total = row['bytes']
    size = _header_size(_range(row, 0, 7, token), total)
    return _header(_range(row, 8, 7 + size, token), size, total, model, parse_config=parse_config)


def _weight_map(value):
    mapping = value.get('weight_map') if isinstance(value, dict) else None
    if not isinstance(mapping, dict) or not mapping or len(mapping) > 100000:
        raise ValueError('Invalid MOSS-Image transformer shard index.')
    for name, file in mapping.items():
        if not name or '/' in hub._safe_path(file) or not file.endswith('.safetensors'):
            raise ValueError('Invalid MOSS-Image transformer shard path.')
    files = set(mapping.values())
    if len({file.casefold() for file in files}) != len(files):
        raise ValueError('MOSS-Image shard filenames must be distinct on all platforms.')
    if len(files) > MAX_SHARDS:
        raise ValueError('Too many MOSS-Image transformer shards to inspect.')
    return mapping


def _moss_shapes(mapping, read):
    by_file = {}
    for name, file in mapping.items():
        by_file.setdefault(file, set()).add(name)
    shapes = {}
    for file, expected in sorted(by_file.items()):
        found, _ = read(file)
        if set(found) != expected:
            raise ValueError('MOSS-Image shard tensor keys disagree with its index.')
        shapes.update(found)
    return shapes, {}


def _selection(model, machine):
    group = GROUPS.get(model) if isinstance(model, str) else None
    if group is None:
        raise ValueError('Choose a native FreeVideo picture model first.')
    if model != image_model.QWEN and machine.get('device_backend') == 'mps':
        raise ValueError('This picture model is unavailable on Apple Silicon.')
    model_root = machine.get('model_root')
    if not model_root or not image_pack.ready(model_root, group=group):
        raise ValueError('Install the official version of this picture model first. Its encoder, decoder and configuration are shared.')
    pack = image_pack.base(model_root)
    rows = image_pack.rows(group=group)
    excluded = set()
    if model == image_model.MOSS:
        path = MOSS_DIRECTORY
        excluded = {row['file'] for row in rows if row['file'].startswith(path + '/')
                    and row['file'].endswith(('.safetensors', '.safetensors.index.json'))}
    else:
        path = next(row['file'] for row in rows if row['component'] == 'dit')
        excluded.add(path)
    companions = [row for row in rows if not row.get('shared') and row['file'] not in excluded]
    return pack, path, companions


def _installed_shapes(pack, path, model):
    root = pack / path
    if model != image_model.MOSS:
        return _local_header(root, model)
    index = io_path(root / INDEX)
    if not index.is_file():
        return _local_header(root / SINGLE, model)
    mapping = _weight_map(hub._json(index.read_bytes()))
    return _moss_shapes(mapping, lambda file: _local_header(root / file, model))


def _choices(entries, prefix, canonical, model):
    if prefix and prefix.endswith(('.safetensors', '.safetensors.index.json')):
        return [prefix]
    prefix = prefix.rstrip('/') + '/' if prefix else ''
    if model == image_model.MOSS:
        choices = [name for name in entries if name.startswith(prefix) and name.rsplit('/', 1)[-1] in (INDEX, SINGLE)]
        return sorted(choices, key=lambda name: (not name.endswith('/transformer/' + INDEX), len(name), name))[:MAX_CHOICES]
    basename = canonical.rsplit('/', 1)[-1]
    choices = []
    for name in entries:
        if not name.startswith(prefix) or not name.endswith('.safetensors'):
            continue
        parts = name[len(prefix):].lower().split('/')
        if any(part in ('text_encoder', 'text_encoders', 'vae', 'loras') for part in parts[:-1]):
            continue
        choices.append(name)
    return sorted(choices, key=lambda name: (name.rsplit('/', 1)[-1] != basename,
                                             'int8_convrot' not in name.lower().replace('-', '_'), len(name), name))[:MAX_CHOICES]


def _qwen_attention_only_difference(found, expected):
    found_tensors, found_config = found
    stock_tensors, stock_config = expected
    return (found_config == stock_config
            and {key: value for key, value in found_tensors.items() if not key.endswith(ATTENTION_HINT)}
            == {key: value for key, value in stock_tensors.items() if not key.endswith(ATTENTION_HINT)})


def _candidate_weights(repo, revision, entries, file, path, model, token):
    def row(name, destination):
        return hub._row(repo, revision, entries, name, 'image_gen/' + destination)

    if model != image_model.MOSS:
        if not file.endswith('.safetensors'):
            raise ValueError('Choose a native image denoiser safetensors file.')
        weight = row(file, path)
        return [weight], _remote_header(weight, model, token)
    if file.endswith('.safetensors'):
        weight = row(file, path + '/' + SINGLE)
        return [weight], _remote_header(weight, model, token)
    if file.rsplit('/', 1)[-1] != INDEX:
        raise ValueError('Choose the MOSS-Image Diffusers transformer shard index.')
    index = row(file, path + '/' + INDEX)
    value, sha = hub._read(repo, revision, entries, file, token)
    index.pop('git_blob', None)
    index['sha256'] = sha
    mapping = _weight_map(value)
    folder = file.rpartition('/')[0]
    shards = {name: row(hub._at(folder, name), path + '/' + name) for name in sorted(set(mapping.values()))}
    shapes = _moss_shapes(mapping, lambda name: _remote_header(shards[name], model, token))
    return [index, *shards.values()], shapes


def _identifier(repo, revision, file, model):
    identity = json.dumps([repo.casefold(), revision, file, model], separators=(',', ':'))
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _space(root, identity, files):
    folder = Path(root) / 'prepared' / 'huggingface-image' / identity
    total = sum(row['bytes'] for row in files)
    remaining = 0
    for row in files:
        target = io_path(folder / row['target'])
        try:
            found = target.is_file() and target.stat().st_size == row['bytes']
        except OSError:
            found = False
        if not found:
            remaining += row['bytes']
    return dict(bytes=total, download_bytes=remaining, required_bytes=remaining + hub.MARGIN)


def inspect(reference, root, machine, model, *, revision=None, subfolder=None, token=None):
    repo, revision, prefix = _reference(reference, revision, subfolder)
    token = hf_auth.validate(token or '')
    pack, path, companions = _selection(model, machine)
    expected = _installed_shapes(pack, path, model)
    info, entries = hub._info(repo, revision, token)
    repo, revision = info['id'], info['sha']
    diagnostic = None
    for file in _choices(entries, prefix, path, model):
        try:
            files, shapes = _candidate_weights(repo, revision, entries, file, path, model, token)
            if shapes != expected:
                if model == image_model.QWEN and _qwen_attention_only_difference(shapes, expected):
                    diagnostic = ('A Qwen INT8 ConvRot checkpoint was found, but its attention metadata differs from '
                                  'the installed Turbo checkpoint. FreeVideo uses the 8-step Turbo profile.')
                    raise ValueError(diagnostic)
                raise ValueError('The image tensor names, shapes or storage types do not match the selected native model.')
        except ValueError:
            if prefix == file:
                raise
        else:
            break
    else:
        if diagnostic:
            raise ValueError(diagnostic)
        folder = prefix.rstrip('/') + '/' if prefix else ''
        if any(name.lower().endswith('.gguf') and (name == prefix or name.startswith(folder)) for name in entries):
            raise ValueError('This selection contains GGUF weights. FreeVideo image switching requires a compatible safetensors denoiser.')
        raise ValueError('No compatible denoiser for the selected picture model was found. Paste the exact checkpoint or shard index URL to inspect it.')
    license_file = 'LICENSE' if 'LICENSE' in entries else None
    if license_file:
        files.append(hub._row(repo, revision, entries, license_file, 'LICENSE'))
    identity = _identifier(repo, revision, file, model)
    return dict(id=identity, repo=repo, revision=revision, path=file, subfolder=prefix or '', label=repo.split('/', 1)[1],
                model=model, kind='image', format=FORMATS[model], weights='image_gen/' + path, files=files,
                base_files=companions, license=hub._license(info), gated=bool(info.get('gated')), compatible=True,
                license_url=hub.HUB + '/' + repo + '/blob/' + revision + '/' + license_file if license_file
                else hub.HUB + '/' + repo + '/tree/' + revision,
                **_space(root, identity, files))


def validate_import(candidate, pack_dir, machine):
    """Recheck on-disk tensor signatures before the native image worker smoke test."""
    model = candidate['model']
    stock, path, _ = _selection(model, machine)
    if candidate.get('weights') != 'image_gen/' + path:
        raise ValueError('The selected image checkpoint does not match the native model path.')
    if _installed_shapes(Path(pack_dir), path, model) != _installed_shapes(stock, path, model):
        raise ValueError('The staged image checkpoint does not match the verified native image model.')
