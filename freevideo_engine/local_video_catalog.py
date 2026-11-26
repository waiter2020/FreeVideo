"""Inspect a locally selected VDN-H3 checkpoint without loading its weights."""
import hashlib
import json
import math
from pathlib import Path

from . import adaln_assets, hf_video_catalog as catalog, prepared_model
from .local_model_copy import source_record
from .long_paths import io_path


TENSOR_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
                'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1}
RAW_DTYPES = {'BF16', 'F16', 'F32'}
MAX_HEADER = 64 * 1024 * 1024


def _source(root, relative, target=None, **expected):
    catalog._safe_path(relative)
    row = source_record(root / relative, target or relative, root=root, **expected)
    if row['bytes'] > 120 * catalog.GIB:
        raise ValueError('Local video model file is too large: ' + relative)
    return row


def _metadata(path, label):
    try:
        with io_path(path).open('rb') as stream:
            data = stream.read(catalog.MAX_METADATA + 1)
    except OSError as error:
        raise ValueError('Cannot read video model metadata: ' + label) from error
    if len(data) > catalog.MAX_METADATA:
        raise ValueError('Video model metadata is too large: ' + label)
    try:
        return json.loads(data, object_pairs_hook=catalog._object), data
    except (ValueError, RecursionError) as error:
        raise ValueError('Invalid video model metadata: ' + label) from error


def _tensor_header(row, *, dtypes=TENSOR_BYTES):
    """Validate bounded header and payload extent without reading tensor data."""
    try:
        with io_path(row['source']).open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            if not 2 <= length <= min(MAX_HEADER, row['bytes'] - 8):
                raise ValueError('Invalid header length')
            value = json.loads(stream.read(length), object_pairs_hook=catalog._object)
        metadata = value.pop('__metadata__', {})
        if not value or not isinstance(metadata, dict):
            raise ValueError('Missing tensor layout')
        layout, spans = {}, []
        for name, tensor in value.items():
            shape, dtype = tensor['shape'], tensor['dtype']
            begin, end = tensor['data_offsets']
            if (not name or dtype not in dtypes or not isinstance(shape, list) or len(shape) > 8
                    or any(type(n) is not int or n < 0 for n in shape)
                    or type(begin) is not int or type(end) is not int or begin < 0
                    or end - begin != math.prod(shape) * TENSOR_BYTES[dtype]):
                raise ValueError('Invalid tensor shape, dtype or offset')
            spans.append((begin, end))
            layout[name] = (dtype, shape)
        cursor = 0
        for begin, end in sorted(spans):
            if begin != cursor:
                raise ValueError('Non-contiguous tensor data')
            cursor = end
        if cursor != row['bytes'] - length - 8:
            raise ValueError('Incomplete tensor data')
        return layout
    except (AttributeError, KeyError, OSError, RecursionError, TypeError, ValueError) as error:
        raise ValueError('Invalid video safetensors: ' + row['target']) from error


def _official_manifest(sha):
    shipped = prepared_model.catalog()
    return any(catalog._official(shipped, shipped['repo'], variant['cache_prefix'], sha)
               for variant in shipped['variants'].values())


def _prepared_headers(manifest, groups, copied):
    dtype, suffix = ('I8', '.weight_int8') if manifest['precision'] == 'int8' else ('F8_E4M3', '.weight_fp8')
    names = set()
    for group in groups:
        row = copied[group['file']]
        layout = _tensor_header(row)
        linears = {name.removesuffix(suffix) for name in layout if name.endswith(suffix)}
        if (('tensors' in group and group['tensors'] != len(layout))
                or ('linears' in group and linears != group['linears'].keys()) or names & linears):
            raise ValueError('Quantized tensor names disagree with the cache manifest: ' + group['file'])
        for name in linears:
            spec = manifest['linears'].get(name)
            if (not spec or layout[name + suffix] != (dtype, spec['weight_shape'])
                    or layout.get(name + '.weight_scale') != ('F32', spec['scale_shape'])):
                raise ValueError('Incompatible quantized tensor shape or dtype: ' + name)
        names.update(linears)
    if names != manifest['linears'].keys():
        raise ValueError('Quantized video tensors are missing from the cache.')


def _prepared(root):
    source = _source(root, 'manifest.json', 'cache/manifest.json')
    manifest, data = _metadata(source['source'], 'manifest.json')
    sha = hashlib.sha256(data).hexdigest()
    source['sha256'] = sha
    format_, slim, groups, tables = catalog._prepared_layout(manifest)
    official = _official_manifest(sha)
    if (manifest.get('adaln_sources') or {}).get('download') and not official:
        raise ValueError('Only pinned official video caches may reference a remote AdaLN source.')
    projections = adaln_assets.projection_groups(manifest) if slim else []
    if official:
        projections = [row for row in projections if io_path(root / row['file']).is_file()]
    records = [*groups, *(row for row, _ in tables), *projections]
    copied = {row['file']: _source(root, row['file'], 'cache/' + row['file'], sha256=row['sha256'], size=row['bytes'])
              for row in records}
    _prepared_headers(manifest, groups, copied)
    for row, identity in tables:
        try:
            adaln_assets.check_table(io_path(copied[row['file']]['source']), row, identity, verify_hash=False)
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError('Invalid AdaLN sampling table: ' + row['file']) from error
    for row in projections:
        _tensor_header(copied[row['file']], dtypes=RAW_DTYPES)
    return dict(kind='prepared', format=format_, base=None, checkpoint=None, cache='cache',
                files=[source, *copied.values()], manifest_sha256=sha, official_prepared=official)


def _raw_configuration(row, machine):
    name = row['target']
    group, short = name.split('/', 1)
    role = 'base' if group == catalog.RAW_BASE else 'checkpoint'
    if not machine.get(role):
        raise ValueError('Repair the installed VDN-H3 configuration before importing source weights.')
    value, data = _metadata(row['source'], name)
    installed, _ = _metadata(Path(machine[role]) / short, 'installed ' + short)
    if value != installed:
        raise ValueError('The source must match the installed VDN-H3 DMD8 model configuration: ' + name)
    if name.endswith('/adapter_config.json'):
        catalog._adapter_config(value)
    row['sha256'] = hashlib.sha256(data).hexdigest()


def _raw(root, machine):
    files = [_source(root, name) for name in catalog._raw_fixed_files()]
    for row in files:
        if row['target'].endswith('.json'):
            _raw_configuration(row, machine)
        else:
            _tensor_header(row, dtypes=RAW_DTYPES)
    expected = {}
    if io_path(root / catalog.RAW_INDEX).exists():
        index = _source(root, catalog.RAW_INDEX)
        value, data = _metadata(index['source'], index['target'])
        index['sha256'] = hashlib.sha256(data).hexdigest()
        files.append(index)
        for key, shard in catalog._raw_weight_map(value).items():
            expected.setdefault(catalog.RAW_TRANSFORMER + '/' + shard, set()).add(key)
    shards = [_source(root, name) for name in (sorted(expected) or [catalog.RAW_SINGLE])]
    for row in shards:
        layout = _tensor_header(row, dtypes=RAW_DTYPES)
        if expected and layout.keys() != expected[row['target']]:
            raise ValueError('Transformer shard tensor names disagree with the index: ' + row['target'])
    files.extend(shards)
    catalog._raw_adapter_headers(root / catalog.RAW_BASE, root / catalog.LAYOUT,
                                 transformer=[io_path(row['source']) for row in shards])
    return dict(kind='raw', format='raw', base=catalog.RAW_BASE, checkpoint=catalog.LAYOUT,
                cache=None, files=files)


def inspect(selection, root, machine):
    """Return a private copy plan for a native prepared cache or raw source folder."""
    selection = Path(selection)
    try:
        if io_path(selection).is_dir():
            directory = selection
            prepared = io_path(selection / 'manifest.json').is_file()
        elif selection.name.casefold() == 'manifest.json':
            directory = selection.parent
            prepared = True
        else:
            raise ValueError('Select a FreeVideo video manifest.json or a complete VDN-H3 source folder.')
        candidate = _prepared(directory) if prepared else _raw(directory, machine)
    except OSError as error:
        raise ValueError('The selected video model files could not be read. Select them again.') from error
    reason = catalog._compatible(candidate['format'], root, machine)
    size = sum(row['bytes'] for row in candidate['files'])
    reserve = catalog.RAW_CACHE_RESERVE if candidate['kind'] == 'raw' else 0
    label = directory.parent.name if prepared and directory.name.casefold() == 'cache' else directory.name
    return dict(candidate, label=label or selection.name, source_kind='local',
                compatible=reason is None, compatibility_reason=reason, bytes=size,
                copy_bytes=size, required_bytes=size + reserve + catalog.MARGIN)


def validate_import(candidate, folder, machine, *, cache=None):
    """Recheck the staged raw package before conversion and use the normal cache checks."""
    if candidate['kind'] == 'raw' and cache is None:
        _raw(Path(folder), machine)
    else:
        catalog.validate_import(candidate, folder, machine, cache=cache)
