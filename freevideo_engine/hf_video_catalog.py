"""Inspect public Hugging Face video packages without fetching their weights."""
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import adaln_assets, hf_auth, prepared_model
from .long_paths import io_path


HUB = 'https://huggingface.co'
REPO = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
REF = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
PART = re.compile(r'[A-Za-z0-9_][A-Za-z0-9._-]{0,127}')
COMMIT = re.compile(r'[0-9a-f]{40}')
SHA256 = re.compile(r'[0-9a-f]{64}')
BLOCKS = 50
LAYOUT = 'stage-dmd-step-250'
RAW_BASE = 'h3-base'
RAW_TRANSFORMER = RAW_BASE + '/transformer'
RAW_SINGLE = RAW_TRANSFORMER + '/diffusion_pytorch_model.safetensors'
RAW_INDEX = RAW_SINGLE + '.index.json'
GIB = 1 << 30
MARGIN = 2 * GIB
# Local conversion outputs 51 groups and 50 full AdaLN projections.
RAW_CACHE_RESERVE = 50 * GIB
MAX_METADATA = 4 * 1024 * 1024
PREPARED_PATHS = ('cache', 'rowwise/cache', 'int8-convrot/cache')


class _HubRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Metadata is loaded from the official API and raw Git paths only.
        if urlsplit(newurl).netloc != 'huggingface.co' or urlsplit(newurl).scheme != 'https':
            raise ValueError('Hugging Face metadata redirected outside the Hub.')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _request(path, *, limit=MAX_METADATA, token=''):
    headers = {'User-Agent': 'FreeVideo/ModelCatalog'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = Request(HUB + path, headers=headers)
    try:
        with build_opener(_HubRedirect()).open(request, timeout=10) as response:
            data = response.read(limit + 1)
    except HTTPError as error:
        if error.code in (401, 403, 404):
            raise ValueError('Hugging Face could not provide the model, commit or file. For gated models, request access on the Hub and add a read token.') from error
        raise ValueError(f'Hugging Face metadata request failed (HTTP {error.code}).') from error
    except (OSError, URLError) as error:
        raise ValueError('Could not reach Hugging Face to inspect the model.') from error
    if len(data) > limit:
        raise ValueError('Hugging Face metadata is too large to inspect.')
    return data


def _object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError('Model metadata contains duplicate fields.')
    return value


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError('Hugging Face returned invalid model metadata.') from error


def _safe_path(value, *, empty=False):
    if value == '' and empty:
        return value
    if (not isinstance(value, str) or len(value) > 512
            or any(not PART.fullmatch(part) or part.endswith('.') or _reserved(part)
                   for part in value.split('/'))):
        raise ValueError('Invalid model file or folder path.')
    return value


def _reserved(name):
    stem = name.split('.')[0].upper()
    return stem in {'CON', 'PRN', 'AUX', 'NUL'} or bool(re.fullmatch(r'(COM|LPT)[0-9]', stem))


def parse_reference(reference, *, revision=None, subfolder=None):
    """Support a repo ID, repo@revision, or a Hugging Face model/tree URL."""
    if not isinstance(reference, str) or not 1 <= len(reference.strip()) <= 1024:
        raise ValueError('Enter a Hugging Face model repository or model URL.')
    reference = reference.strip()
    embedded_ref = embedded_path = None
    if '://' in reference:
        url = urlsplit(reference)
        if (url.scheme != 'https' or url.netloc != 'huggingface.co' or url.query or url.fragment):
            raise ValueError('Enter an HTTPS model URL on huggingface.co.')
        path = unquote(url.path.strip('/')).split('/')
        reference = '/'.join(path[:2])
        if len(path) > 2:
            if len(path) < 4 or path[2] != 'tree':
                raise ValueError('Use a Hugging Face model repository or tree URL.')
            embedded_ref = path[3]
            embedded_path = '/'.join(path[4:]) or None
    elif '@' in reference:
        reference, embedded_ref = reference.split('@', 1)
    if not REPO.fullmatch(reference) or '..' in reference or any(part.endswith('.') for part in reference.split('/')):
        raise ValueError('Use the Hugging Face repository ID owner/model.')
    if (revision is not None and embedded_ref is not None and revision != embedded_ref
            or subfolder is not None and embedded_path is not None and subfolder != embedded_path):
        raise ValueError('The repository URL and model selection disagree.')
    revision = revision if revision is not None else embedded_ref
    subfolder = subfolder if subfolder is not None else embedded_path
    if revision is not None and (not isinstance(revision, str) or not REF.fullmatch(revision) or '..' in revision):
        raise ValueError('Invalid Hugging Face commit, tag or branch.')
    if subfolder is not None:
        _safe_path(subfolder, empty=True)
    return reference, revision, subfolder


def _license(info):
    card = info.get('cardData') or {}
    if isinstance(card, dict):
        name = card.get('license_name') or card.get('license')
        if isinstance(name, str):
            return name[:160]
    tags = info.get('tags')
    return next((tag.removeprefix('license:')[:160] for tag in tags
                 if isinstance(tag, str) and tag.startswith('license:')), 'unspecified') if isinstance(tags, list) else 'unspecified'


def search(query, *, token=None):
    """Find public candidates. An exact compatibility check happens in inspect()."""
    if not isinstance(query, str) or not 2 <= len(query.strip()) <= 120:
        raise ValueError('Enter 2–120 characters to search Hugging Face.')
    token = hf_auth.validate(token or '')
    params = {'search': query.strip(), 'limit': 20, 'full': 'true'}
    if not token:
        params['gated'] = 'false'
    result = _json(_request('/api/models?' + urlencode(params), token=token))
    if not isinstance(result, list):
        raise ValueError('Invalid Hugging Face search response.')
    return [dict(repo=row['id'], license=_license(row)) for row in result[:20]
            if isinstance(row, dict) and REPO.fullmatch(str(row.get('id', '')))
            and not row.get('private') and (token or not row.get('gated'))]


def _info(repo, requested, token):
    endpoint = '/api/models/' + repo
    if requested:
        endpoint += '/revision/' + quote(requested, safe='')
    info = _json(_request(endpoint + '?blobs=true', token=token))
    if (not isinstance(info, dict) or not info.get('sha') or not COMMIT.fullmatch(str(info['sha']))
            or not REPO.fullmatch(str(info.get('id', ''))) or info['id'].casefold() != repo.casefold()):
        raise ValueError('Hugging Face did not return a pinned model commit.')
    if info.get('private') or info.get('disabled'):
        raise ValueError('Public and public gated Hugging Face models are supported.')
    if info.get('gated'):
        if not token:
            raise ValueError('Request access to this gated model on Hugging Face and add a read token.')
        _request('/api/models/' + repo + '/auth-check', token=token)
    if not isinstance(info.get('siblings'), list):
        raise ValueError('Hugging Face did not return a model file list.')
    entries = {}
    for row in info['siblings']:
        name = row.get('rfilename') if isinstance(row, dict) else None
        if not isinstance(name, str) or name in entries:
            raise ValueError('Hugging Face returned an invalid model file list.')
        entries[name] = row
    return info, entries


def _at(prefix, file):
    return prefix + '/' + file if prefix else file


def _row(repo, revision, entries, file, target, *, manifest_row=None):
    _safe_path(file)
    _safe_path(target)
    metadata = entries.get(file) or {}
    size = metadata.get('size')
    if type(size) is not int or not 0 < size <= 120 * GIB:
        raise ValueError('Hugging Face file is missing or has an invalid size: ' + file)
    result = dict(repo=repo, revision=revision, file=file, target=target, bytes=size, official_only=True)
    lfs = metadata.get('lfs') or {}
    if not isinstance(lfs, dict):
        raise ValueError('Invalid Hugging Face file metadata: ' + file)
    sha = lfs.get('sha256')
    if sha and SHA256.fullmatch(str(sha)) and lfs.get('size') == size:
        result['sha256'] = sha
    elif not file.endswith('.safetensors') and COMMIT.fullmatch(str(metadata.get('blobId', ''))):
        result['git_blob'] = metadata['blobId']
    else:
        raise ValueError('Hugging Face file has no usable content hash: ' + file)
    if manifest_row and (manifest_row['bytes'] != size or manifest_row['sha256'] != result.get('sha256')):
        raise ValueError('The prepared manifest disagrees with the Hub file metadata: ' + file)
    return result


def _read(repo, revision, entries, file, token):
    expected = _row(repo, revision, entries, file, file)
    if expected['bytes'] > MAX_METADATA:
        raise ValueError('Prepared model metadata file is too large: ' + file)
    data = _request('/' + repo + '/raw/' + revision + '/' + quote(file, safe='/'), token=token)
    if not _matches(data, expected):
        raise ValueError('Hugging Face metadata failed verification: ' + file)
    return _json(data), hashlib.sha256(data).hexdigest()


def _matches(data, expected):
    if len(data) != expected['bytes']:
        return False
    if expected.get('git_blob'):
        blob = hashlib.sha1(f'blob {len(data)}\x00'.encode() + data, usedforsecurity=False).hexdigest()
        return blob == expected['git_blob']
    return hashlib.sha256(data).hexdigest() == expected['sha256']


def _hardware(root, machine):
    report = {}
    try:
        report = json.loads(io_path(Path(root) / 'kernel-capabilities.json').read_bytes())
    except (OSError, ValueError):
        pass
    native = machine.get('device_backend') == 'mps'
    hardware = machine.get('hardware') or report.get('hardware') or {}
    capability = hardware.get('capability') or ()
    if not isinstance(capability, (list, tuple)) or len(capability) != 2 or any(type(v) is not int for v in capability):
        capability = ()
    return native, tuple(capability)


def _compatible(format_, root, machine):
    native, capability = _hardware(root, machine)
    if native:
        return None if format_ == 'int8_convrot' else 'Apple Silicon requires a prepared ConvRot INT8 model.'
    if capability < (8, 0):
        return 'The installed NVIDIA GPU is unknown or does not meet FreeVideo requirements.'
    if format_ == 'fp8_per_tensor' and (8, 9) <= capability < (10, 0):
        return 'This GPU uses the rowwise FP8 format.'
    if format_ == 'fp8_rowwise' and capability >= (10, 0):
        return 'This GPU uses the per-tensor FP8 format.'
    return None


def _preferred(root, machine):
    native, capability = _hardware(root, machine)
    if native or machine.get('prepared_format') == 'int8_convrot':
        return 'int8_convrot'
    return 'fp8_per_tensor' if capability >= (10, 0) else 'fp8_rowwise'


def _record(row):
    if (not isinstance(row, dict) or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 120 * GIB
            or not SHA256.fullmatch(str(row.get('sha256', '')))):
        raise ValueError('Invalid prepared model file record.')


def _prepared_layout(manifest):
    try:
        if manifest.get('online_lora', {}) != {}:
            raise ValueError('Select a base FreeVideo cache without embedded LoRA runtime files.')
        precision = manifest['precision']
        granularity = manifest['scale_granularity']
        layout = manifest.get('format')
        rotation = manifest.get('rotation') or {}
        if precision == 'fp8' and granularity in ('rowwise', 'per_tensor') and layout in (None, adaln_assets.SLIM_FORMAT):
            format_ = 'fp8_' + granularity
        elif (precision == 'int8' and granularity == 'rowwise' and layout == 'freevideo-int8-slim-v1'
              and rotation.get('kind') == 'convrot' and rotation.get('group') == 256):
            format_ = 'int8_convrot'
        else:
            raise ValueError('Use a FreeVideo FP8 cache or a ConvRot INT8 cache.')
        slim = layout in adaln_assets.SLIM_FORMATS
        expected = {'root', *(f'blocks/{i:02d}' for i in range(BLOCKS))}
        if not slim:
            expected.update(f'adaln/{i:02d}' for i in range(BLOCKS))
        groups = manifest['groups']
        if len(groups) != len(expected) or {row['group'] for row in groups} != expected:
            raise ValueError('Expected a complete 50-block VDN-H3 cache.')
        if not SHA256.fullmatch(str(manifest['source_id'])):
            raise ValueError('Prepared model has no valid source identity.')
        for row in groups:
            _record(row)
            if row['file'] != row['group'] + '.safetensors':
                raise ValueError('Invalid prepared model group path.')
        linears = manifest['linears']
        if not isinstance(linears, dict) or len(linears) != 363:
            raise ValueError('Expected 363 quantized VDN-H3 projections.')
        for name, row in linears.items():
            shape = row['weight_shape']
            if (not re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', name)
                    or not isinstance(shape, list) or len(shape) != 2
                    or any(type(n) is not int or not 1 <= n <= 65536 for n in shape)
                    or row['scale_shape'] != [1, 1 if granularity == 'per_tensor' else shape[0]]
                    or row.get('input_dtype') != 'bfloat16'):
                raise ValueError('Invalid quantized VDN-H3 projection.')
            if format_ == 'int8_convrot' and (row.get('storage') != 'int8' or row.get('convrot_group') != 256
                                              or shape[1] % 256):
                raise ValueError('Invalid ConvRot INT8 projection.')
        grouped = {}
        for group in groups:
            entries = group.get('linears', {})
            if not isinstance(entries, dict) or grouped.keys() & entries.keys() or any(
                    spec != linears.get(name) for name, spec in entries.items()):
                raise ValueError('Quantized projections disagree with their prepared groups.')
            grouped.update(entries)
        if all('linears' in group for group in groups) and grouped != linears:
            raise ValueError('Prepared groups do not cover every quantized projection.')
        tables = list(adaln_assets.validate_catalog(manifest, BLOCKS))
        for row, table in tables:
            _record(row)
            if table['channels'] != 5376:
                raise ValueError('Invalid VDN-H3 AdaLN channel count.')
        return format_, slim, groups, tables
    except (AttributeError, KeyError, TypeError, OverflowError) as error:
        raise ValueError('Invalid FreeVideo prepared model manifest.') from error


def _official(catalog, repo, prefix, sha):
    if repo != catalog['repo']:
        return False
    return any(variant['cache_prefix'] == prefix and any(
        row['file'] == _at(prefix, 'manifest.json') and row['sha256'] == sha for row in variant['files'])
        for variant in catalog['variants'].values())


def _prepared(repo, revision, entries, prefix, catalog, token):
    # Command and restore code use paths inside the local profile, independent of the Hub folder name.
    manifest_file = _at(prefix, 'manifest.json')
    manifest, sha = _read(repo, revision, entries, manifest_file, token)
    format_, slim, groups, tables = _prepared_layout(manifest)
    official = _official(catalog, repo, prefix, sha)
    remote = (manifest.get('adaln_sources') or {}).get('download')
    if remote and not official:
        raise ValueError('Community packages must include their full AdaLN projections in the cache.')
    records = [*groups, *(row for row, _ in tables)]
    if slim and not official:
        records.extend(adaln_assets.projection_groups(manifest))
    files = [_row(repo, revision, entries, manifest_file, 'cache/manifest.json')]
    files[0].pop('git_blob', None)
    files[0]['sha256'] = sha
    files.extend(_row(repo, revision, entries, _at(prefix, row['file']), 'cache/' + row['file'], manifest_row=row)
                 for row in records)
    return dict(kind='prepared', format=format_, base=None, checkpoint=None, cache='cache',
                files=files, manifest_sha256=sha, official_prepared=official)


def _adapter_config(value):
    """The existing raw merger supports the declared B @ A update at unit scale."""
    config = value.get('config') if isinstance(value, dict) and value.get('type') == 'lora' else None
    if not isinstance(config, dict) or value.get('version') != 1:
        raise ValueError('Expected a native VDN LoRA adapter configuration.')
    rank, alpha = config.get('rank'), config.get('alpha')
    ranks, alphas = config.get('rank_pattern', {}), config.get('alpha_pattern', {})
    targets = config.get('targets')
    if (type(rank) is not int or rank < 1 or type(alpha) is not int or alpha != rank
            or not isinstance(ranks, dict) or not isinstance(alphas, dict)
            or any(type(item) is not int or item < 1 for item in (*ranks.values(), *alphas.values()))
            or any(ranks.get(target, rank) != alphas.get(target, alpha) for target in ranks.keys() | alphas.keys())
            or not isinstance(targets, list) or not targets
            or any(not isinstance(target, str) or not target for target in targets)):
        raise ValueError('The VDN LoRA must use equal alpha and rank for every target.')
    return config


def _adapter_shapes(shapes, source_shapes, adapter, config):
    patterns = config['targets']
    if config.get('exact_targets'):
        targets = set(patterns)
    else:
        targets = {name for name in source_shapes if any(
            fnmatchcase(name, pattern) or fnmatchcase(name, '*.' + pattern) for pattern in patterns)}
    ranks = config.get('rank_pattern', {})
    expected = {}
    if not targets or set(ranks) - targets:
        raise ValueError('The VDN LoRA declares unsupported targets: ' + adapter)
    for target in targets:
        shape = source_shapes.get(target)
        if not shape or len(shape) != 2:
            raise ValueError('A VDN LoRA target is missing or is not a matrix: ' + target)
        rank = ranks.get(target, config['rank'])
        expected[target + '.lora_A.' + adapter + '.weight'] = (rank, shape[1])
        expected[target + '.lora_B.' + adapter + '.weight'] = (shape[0], rank)
    if shapes != expected:
        raise ValueError('VDN LoRA tensor names or shapes disagree with the declared targets and ranks: ' + adapter)


def _raw_adapter_headers(base, checkpoint, *, transformer=None):
    from .lora_check import read_header
    if transformer is None:
        transformer = sorted(io_path(base / 'transformer').glob('*.safetensors'))
    source_shapes = {}
    for file in (*transformer, io_path(checkpoint / 'linear_branch/model.safetensors')):
        shapes, _ = read_header(file)
        for name, shape in shapes.items():
            if not name.endswith('.weight') or len(shape) != 2:
                continue
            if file in transformer:
                name = re.sub(r'^(transformer_blocks\.\d+\.attn)\.', r'\1.orig.', name)
            target = name.removesuffix('.weight')
            if target in source_shapes:
                raise ValueError('Duplicate raw VDN-H3 source tensor: ' + name)
            source_shapes[target] = shape
    for adapter in ('default', 'turbo'):
        directory = io_path(checkpoint / 'adapters' / adapter)
        config = _adapter_config(_json((directory / 'adapter_config.json').read_bytes()))
        shapes, _ = read_header(directory / 'adapter_model.safetensors')
        _adapter_shapes(shapes, source_shapes, adapter, config)


def _raw_fixed_files():
    canonical = json.loads(Path(__file__).with_name('model_files.json').read_bytes())
    fixed = [row['file'] for row in canonical if row['file'].startswith((
        RAW_TRANSFORMER + '/', LAYOUT + '/adapters/', LAYOUT + '/linear_branch/'))
        and row['file'] != RAW_INDEX and not row['file'].startswith(RAW_TRANSFORMER + '/diffusion_pytorch_model-')]
    fixed.append(LAYOUT + '/model_spec.json')
    return fixed


def _raw_weight_map(index):
    mapping = index.get('weight_map') if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or not mapping or len(mapping) > 100000:
        raise ValueError('Invalid VDN-H3 transformer shard index.')
    for name in mapping.values():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', name)
                or _reserved(name)):
            raise ValueError('Invalid VDN-H3 transformer shard name.')
    names = set(mapping.values())
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError('VDN-H3 shard filenames must be distinct on all platforms.')
    return mapping


def _raw(repo, revision, entries, prefix, machine, token):
    fixed = _raw_fixed_files()
    configurations = [name for name in fixed if name.endswith('.json')]
    for relative in configurations:
        role = 'base' if relative.startswith(RAW_BASE + '/') else 'checkpoint'
        installed = Path(machine[role]) / relative.split('/', 1)[1]
        try:
            current = io_path(installed).read_bytes()
        except OSError as error:
            raise ValueError('The installed VDN-H3 model configuration is missing. Repair it before importing source weights.') from error
        remote = _at(prefix, relative)
        value = _json(current)
        if not _matches(current, _row(repo, revision, entries, remote, relative)) and value != _read(
                repo, revision, entries, remote, token)[0]:
            raise ValueError('The source must match the installed VDN-H3 DMD8 model configuration: ' + relative)
        if relative.endswith('/adapter_config.json'):
            _adapter_config(value)
    if _at(prefix, RAW_INDEX) in entries:
        index, _ = _read(repo, revision, entries, _at(prefix, RAW_INDEX), token)
        mapping = _raw_weight_map(index)
        fixed.append(RAW_INDEX)
        fixed.extend(RAW_TRANSFORMER + '/' + name for name in sorted(set(mapping.values())))
    else:
        fixed.append(RAW_SINGLE)
    files = [_row(repo, revision, entries, _at(prefix, name), name) for name in fixed]
    return dict(kind='raw', format='raw', base=RAW_BASE, checkpoint=LAYOUT, cache=None, files=files)


def _choices(entries):
    choices = list(PREPARED_PATHS) + ['']
    for name in entries:
        if name.endswith('/manifest.json'):
            prefix = name.removesuffix('/manifest.json')
            if prefix not in choices and _at(prefix, 'blocks/00.safetensors') in entries:
                try:
                    choices.append(_safe_path(prefix))
                except ValueError:
                    continue
    return [prefix for prefix in choices if _at(prefix, 'manifest.json') in entries][:8]


def _identifier(repo, revision, prefix, format_):
    return hashlib.sha256(json.dumps([repo.casefold(), revision, prefix, format_]).encode()).hexdigest()[:24]


def _space(root, identifier, files, kind):
    folder = Path(root) / 'prepared' / 'huggingface' / identifier
    total = sum(row['bytes'] for row in files)
    remaining = 0
    for row in files:
        target = folder / row['target']
        try:
            path = io_path(target)
            if path.is_file() and path.stat().st_size == row['bytes']:
                continue
        except OSError:
            pass
        remaining += row['bytes']
    reserve = RAW_CACHE_RESERVE if kind == 'raw' else 0
    return dict(bytes=total, download_bytes=remaining, required_bytes=remaining + reserve + MARGIN)


def inspect(reference, root, machine, *, revision=None, subfolder=None, token=None):
    repo, requested, subfolder = parse_reference(reference, revision=revision, subfolder=subfolder)
    token = hf_auth.validate(token or '')
    catalog = prepared_model.catalog()
    preferred = _preferred(root, machine)
    # Use the built-in immutable pin for the official entry with no explicit revision.
    if repo.casefold() == catalog['repo'].casefold() and requested is None:
        choice = {'fp8_per_tensor': 'per_tensor', 'fp8_rowwise': 'rowwise', 'int8_convrot': 'int8_convrot'}[preferred]
        variant = next((row for row in catalog['variants'].values() if row['cache_prefix'] == subfolder),
                       catalog['variants'][choice])
        requested = variant.get('revision', catalog['revision'])
    info, entries = _info(repo, requested, token)
    repo, revision = info['id'], info['sha']
    choices = [subfolder] if subfolder is not None else _choices(entries)
    prepared = []
    errors = []
    for prefix in choices:
        if _at(prefix, 'manifest.json') not in entries:
            continue
        try:
            candidate = _prepared(repo, revision, entries, prefix, catalog, token)
        except ValueError as error:
            errors.append(error)
        else:
            prepared.append((prefix, candidate))
    if prepared:
        prefix, candidate = min(prepared, key=lambda item: (
            bool(_compatible(item[1]['format'], root, machine)), item[1]['format'] != preferred))
    else:
        prefix = subfolder or ''
        if _at(prefix, LAYOUT + '/model_spec.json') not in entries:
            if errors:
                raise errors[0]
            raise ValueError('No FreeVideo prepared cache or VDN-H3 DMD8 source layout was found in this repository.')
        candidate = _raw(repo, revision, entries, prefix, machine, token)
    format_ = candidate['format']
    identifier = _identifier(repo, revision, prefix, format_)
    reason = _compatible(format_, root, machine)
    label = repo.split('/', 1)[1]
    license_file = 'LICENSE' if 'LICENSE' in entries else None
    if license_file:
        candidate['files'].append(_row(repo, revision, entries, license_file, 'LICENSE'))
    variants = [dict(path=path, subfolder=path, revision=revision, format=value['format'],
                     compatible=_compatible(value['format'], root, machine) is None)
                for path, value in prepared]
    if repo == catalog['repo'] and not subfolder:
        for name, variant in catalog['variants'].items():
            path = variant['cache_prefix']
            if not any(row['path'] == path for row in variants):
                label_format = 'int8_convrot' if name == 'int8_convrot' else 'fp8_' + name
                variants.append(dict(path=path, subfolder=path, revision=variant.get('revision', catalog['revision']),
                                     format=label_format, compatible=_compatible(label_format, root, machine) is None))
    return dict(candidate, id=identifier, repo=repo, revision=revision, path=prefix, subfolder=prefix, label=label,
                license=_license(info), license_url=HUB + '/' + repo + '/blob/' + revision + '/' + license_file
                if license_file else HUB + '/' + repo + '/tree/' + revision,
                compatible=reason is None, compatibility_reason=reason, variants=variants, gated=bool(info.get('gated')),
                **_space(root, identifier, candidate['files'], candidate['kind']))


def validate_import(candidate, folder, machine, *, cache=None):
    """Check local metadata before provisioning validates every tensor and runs inference."""
    folder = Path(folder)
    if candidate['kind'] == 'prepared' and cache is None:
        cache = folder / candidate['cache']
    if cache is not None:
        data = io_path(Path(cache) / 'manifest.json').read_bytes()
        format_, _, _, _ = _prepared_layout(_json(data))
        if candidate['kind'] == 'prepared' and (format_ != candidate['format'] or
                hashlib.sha256(data).hexdigest() != candidate['manifest_sha256']):
            raise ValueError('The model manifest changed after inspection.')
        if candidate['kind'] == 'raw' and not format_.startswith('fp8_'):
            raise ValueError('VDN source conversion did not produce an FP8 cache.')
    if candidate['kind'] == 'raw':
        for key, relative in (('base', 'transformer/config.json'), ('checkpoint', 'model_spec.json')):
            if _json(io_path(folder / candidate[key] / relative).read_bytes()) != _json(io_path(Path(machine[key]) / relative).read_bytes()):
                raise ValueError('The source model configuration changed after inspection.')
        if cache is None:
            _raw_adapter_headers(folder / candidate['base'], folder / candidate['checkpoint'])
