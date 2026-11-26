"""Inspect and copy files the operator selected on this computer."""
import hashlib
import json
import os
from pathlib import Path
import stat

from .adaln_assets import asset_path
from .file_ops import publish
from .local_models import BLOCK, discard_read_cache
from .long_paths import io_path, plain, resolved
from .storage import fingerprint


def _source(path, root):
    root, path = Path(root), Path(path)
    try:
        relative = path.relative_to(root)
        current = root
        if Path(plain(resolved(root))) != root:
            raise ValueError
        for part in relative.parts:
            current /= part
            entry = io_path(current).lstat()
            if stat.S_ISLNK(entry.st_mode) or getattr(entry, 'st_file_attributes', 0) & 0x400:
                raise ValueError
        if not stat.S_ISREG(io_path(path).stat().st_mode):
            raise ValueError
        return path
    except (ValueError, OSError) as error:
        raise ValueError(f'Missing or unsafe local model file: {path.name}') from error


def source_record(source, target, *, root, sha256=None, size=None):
    """Keep filesystem access details in the private pending selection."""
    root = Path(plain(resolved(root)))
    source = Path(plain(io_path(source).absolute()))
    source = _source(source, root)
    stamp = fingerprint(source)
    if not stamp['bytes'] or (size is not None and stamp['bytes'] != size):
        raise ValueError(f'Unexpected size of local model file: {source.name}')
    row = dict(source=str(source), root=str(root), target=target, bytes=stamp['bytes'], fingerprint=stamp)
    if sha256 is not None:
        row['sha256'] = sha256
    return row


def copy_files(folder, files, progress):
    """Copy and hash in one read, publishing only complete, flushed files."""
    copied, hashes = 0, []
    for row in files:
        progress(copied)
        source = _source(row['source'], row['root'])
        target = io_path(asset_path(folder, row['target']))
        expected = row['fingerprint']
        if fingerprint(source) != expected:
            raise ValueError(f'Local model file changed. Select it again: {source.name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.copying')
        digest = hashlib.sha256()
        done = 0
        with io_path(source).open('rb') as incoming, temporary.open('xb') as outgoing:
            info = os.fstat(incoming.fileno())
            if (info.st_dev, info.st_ino, info.st_size) != (expected['device'], expected['inode'], row['bytes']):
                raise ValueError(f'Local model file changed. Select it again: {source.name}')
            for block in iter(lambda: incoming.read(BLOCK), b''):
                done += len(block)
                if done > row['bytes']:
                    raise ValueError(f'Local model file grew while being copied: {source.name}')
                outgoing.write(block)
                digest.update(block)
                discard_read_cache(incoming.fileno(), done, len(block))
                progress(copied + done)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if done != row['bytes'] or fingerprint(_source(source, row['root'])) != expected:
            raise ValueError(f'Local model file changed during copying: {source.name}')
        checksum = digest.hexdigest()
        if row.get('sha256', checksum) != checksum:
            raise ValueError(f'Local model file failed its SHA-256 check: {source.name}')
        publish(temporary, target)
        copied += done
        hashes.append((row['target'], done, checksum))
    return sorted(hashes)


def content_id(media, candidate, hashes, *, attempt=0):
    value = ['local', media, candidate.get('model'), candidate['kind'], candidate['format'], hashes, attempt]
    if 'component' in candidate:
        value.append(candidate['component'])
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()[:24]
