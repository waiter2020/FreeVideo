"""CPU checks for input-cache ownership, eviction, and cross-process publication."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from freevideo_engine import input_cache, private_store


class InputCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / 'input-cache'
        self.cache = input_cache.InputCache({'test': 'cache'}, root=self.base)
        free = patch.object(input_cache.disk_space, 'free_bytes', return_value=2**32)
        free.start()
        self.addCleanup(free.stop)

    def put(self, key, value):
        source = self.root / 'source.pt'
        source.write_bytes(value)
        self.assertTrue(self.cache.put(key, source, {'success': True}))
        return self.cache.lookup(key)[0]

    def previous(self, key):
        receipt = self.cache.receipt(key)
        path = receipt.with_name(f'{receipt.stem[:38]}-{"a" * 16}.previous.json')
        path.write_bytes(receipt.read_bytes())
        return path

    def test_capture_is_request_local_and_restored_after_error(self):
        first = self.cache.receipt('first')
        second = self.cache.receipt('second')
        with patch.object(input_cache, 'USED', []):
            input_cache._note(first)
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                with input_cache.capture_used() as taken:
                    input_cache._note(second)
                    input_cache._note(second)
                    self.assertEqual(input_cache.used(), [second.relative_to(self.base).as_posix()])
                    raise RuntimeError('cancelled')
            self.assertEqual(len(taken), 1)
            self.assertEqual(input_cache.used(), [first.relative_to(self.base).as_posix()])
            with input_cache.capture_used() as next_request:
                self.assertEqual(next_request, [])

    def test_replacement_keeps_shared_payload_then_reclaims_it(self):
        old = self.put('first', b'old')
        self.assertEqual(self.put('second', b'old'), old)
        archive = self.previous('first')
        self.put('first', b'new-first')
        self.assertTrue(old.is_file())
        self.assertTrue(archive.is_file())
        self.put('second', b'new-second')
        self.assertFalse(old.exists())
        self.assertFalse(archive.exists())
        self.put('second', b'new-second')
        self.assertEqual(list(self.cache.root.glob('*.previous.json')), [])

    def test_eviction_from_an_old_read_keeps_replaced_receipt(self):
        self.put('first', b'old')
        previous = self.cache.lookup('first')[1]
        new = self.put('first', b'new')
        self.cache.evict(self.cache.receipt('first'), expected=previous)
        self.assertEqual(self.cache.lookup('first')[0], new)

    def test_moved_source_stays_when_larger_than_the_cache_budget(self):
        source = self.root / 'move.pt'
        source.write_bytes(b'candidate')
        with patch.object(input_cache, 'limit_bytes', return_value=1):
            with input_cache.capture_used() as taken:
                self.assertFalse(self.cache.put('oversized', source, {'success': True}, move=True))
                self.assertEqual(taken, [])
        self.assertTrue(source.is_file())
        self.assertTrue(self.cache.put('fits', source, {'success': True}, move=True))
        self.assertFalse(source.exists())
        self.assertIsNotNone(self.cache.lookup('fits'))

    def test_budget_includes_archived_orphans_and_retains_newest_active(self):
        first = self.put('first', b'first')
        second = self.put('second', b'second')
        first_receipt = self.cache.receipt('first')
        second_receipt = self.cache.receipt('second')
        os.utime(first_receipt, (1, 1))
        archive = self.previous('second')
        orphan = self.cache.root / ('e' * 64 + '.pt')
        orphan.write_bytes(b'orphan')
        unrelated = self.cache.root / 'user-notes.txt'
        unrelated.write_text('kept', encoding='utf-8')

        self.cache.prune(max_bytes=second_receipt.stat().st_size + second.stat().st_size)
        self.assertFalse(first.exists())
        self.assertFalse(first_receipt.exists())
        self.assertFalse(archive.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue(second.is_file())
        self.assertTrue(second_receipt.is_file())
        self.assertTrue(unrelated.is_file())
        self.cache.prune(max_bytes=0)
        self.assertEqual(private_store.cache_usage([self.base]), {'files': 0, 'bytes': 0})
        self.assertTrue(unrelated.is_file())

    def test_explicit_eviction_retires_archive_references_safely(self):
        current = self.put('first', b'current')
        old = self.put('second', b'old')
        archived = self.previous('second')
        active_first = self.cache.receipt('first')
        active_second = self.cache.receipt('second')
        old_receipt = json.loads(active_second.read_text(encoding='utf-8'))
        another_archive = active_first.with_name(f'{active_first.stem[:38]}-{"b" * 16}.previous.json')
        another_archive.write_text(json.dumps(old_receipt), encoding='utf-8')

        self.cache.evict(active_second)
        self.assertFalse(active_second.exists())
        self.assertFalse(archived.exists())
        self.assertTrue(old.is_file())
        self.cache.evict(active_first)
        self.assertFalse(another_archive.exists())
        self.assertFalse(old.exists())
        self.assertFalse(current.exists())

    def test_plaintext_purge_reaches_orphaned_legacy_payloads(self):
        encoded = self.put('first', b'current')
        plaintext = self.cache.root / ('d' * 64 + '.pt')
        with ZipFile(plaintext, 'w') as archive:
            archive.writestr('data.pkl', input_cache.PLAIN_PROMPT[0])
        orphan = self.cache.root / ('e' * 64 + '.pt')
        orphan.write_bytes(plaintext.read_bytes())
        legacy = self.previous('first')
        legacy.write_text(json.dumps({'file': plaintext.name}), encoding='utf-8')

        self.assertEqual(private_store.purge_prompt_text(self.base), 2)
        self.assertFalse(plaintext.exists())
        self.assertFalse(orphan.exists())
        self.assertFalse(legacy.exists())
        self.assertTrue(encoded.is_file())
        self.assertIsNotNone(self.cache.lookup('first'))

    def test_clear_waits_for_another_process_to_publish(self):
        self.put('existing', b'older')
        unrelated = self.cache.root / 'keep.txt'
        unrelated.write_text('kept', encoding='utf-8')
        ready = self.root / 'publishing'
        release = self.root / 'release'
        child_script = """
import sys, time
from pathlib import Path
from unittest.mock import patch
from freevideo_engine import input_cache
base, ready, release = map(Path, sys.argv[1:])
source = base.parent / 'child.pt'
source.write_bytes(b'new')
cache = input_cache.InputCache({'test': 'cache'}, root=base)
save = input_cache.save
def pause(*args, **kwargs):
    ready.touch()
    deadline = time.monotonic() + 5
    while not release.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('publication was not released')
        time.sleep(.02)
    return save(*args, **kwargs)
with patch.object(input_cache, 'save', pause), patch.object(input_cache.disk_space, 'free_bytes', return_value=2**32):
    assert cache.put('child', source, {'success': True})
"""
        cmd = [sys.executable, '-B', '-c', child_script, str(self.base), str(ready), str(release)]
        repo = Path(__file__).resolve().parents[1]
        with subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
            try:
                deadline = time.monotonic() + 5
                while not ready.is_file():
                    self.assertIsNone(child.poll())
                    self.assertLess(time.monotonic(), deadline, 'child never started publication')
                    time.sleep(.02)
                with ThreadPoolExecutor(max_workers=1) as thread:
                    cleared = thread.submit(private_store.clear_cache, [self.base])
                    try:
                        with self.assertRaises(TimeoutError):
                            cleared.result(timeout=.15)
                    finally:
                        release.touch()
                    self.assertGreater(cleared.result(timeout=4), 0)
                out, error = child.communicate(timeout=4)
                self.assertEqual(child.returncode, 0, out + error)
            finally:
                if child.poll() is None:
                    release.touch()
                    child.kill()
        self.assertIsNone(self.cache.lookup('existing'))
        self.assertIsNone(self.cache.lookup('child'))
        self.assertEqual(private_store.cache_usage([self.base]), {'files': 0, 'bytes': 0})
        self.assertTrue(unrelated.is_file())
        self.put('after clear', b'rebuilt')


if __name__ == '__main__':
    unittest.main()
