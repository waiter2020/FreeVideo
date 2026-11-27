import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from freevideo_engine.launcher_pi_h3 import PiH3Downloads


def translate(en, _zh, values=None):
    return en.format(**(values or {}))


class LauncherPiH3Test(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        self.engine = directory / 'selected'
        self.engine.mkdir()
        self.root = directory / 'library'
        (self.engine / 'machine.json').write_text(json.dumps(dict(ready=True, model_root=str(self.root))), encoding='utf-8')
        self.enterContext(patch.dict(os.environ, FREEVIDEO_MODEL_ROOT=''))
        self.backend = Mock()
        self.backend.owns_download.return_value = False
        self.values = {mode: dict(compatible=True, published=True, present=False, bytes=1000,
                                  resume_bytes=0, install={}) for mode in ('fast2', 'fast4')}
        self.backend.status.side_effect = self.values.__getitem__
        self.factory = self.enterContext(patch('freevideo_engine.launcher_pi_h3.fast_pack.Installer', return_value=self.backend))
        self.manager = PiH3Downloads()
        self.addCleanup(self.manager.close)

    def poll(self):
        self.manager.poll(self.engine)
        thread = self.manager.thread
        if thread is not None:
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.manager.poll(self.engine)

    def rows(self):
        return {row['mode']: row for row in self.manager.view(translate, can_start=True)['rows']}

    def test_selected_installation_works_without_a_server(self):
        self.values['fast2'].update(present=True, bytes=0)
        self.values['fast4'].update(compatible=False, compatibility_reason='gpu')
        self.poll()
        args, kwargs = self.factory.call_args
        self.assertEqual(args[0](), self.root)
        self.assertEqual(kwargs['engine_root'], self.engine)
        self.assertEqual(self.backend.status.call_args_list[0].args, ('fast2',))
        rows = self.rows()
        self.assertEqual(rows['fast2']['title'], 'pi-H3 · 2 steps')
        self.assertEqual(rows['fast2']['detail'], 'Installed')
        self.assertEqual(rows['fast4']['detail'], 'Requires an NVIDIA CUDA GPU')
        self.assertFalse(rows['fast4']['show_download'])

    def test_verification_progress_and_cancel_use_the_observed_job(self):
        self.values['fast4']['install'] = dict(state='running', job_id='job', phase='verify', verify_done=3, verify_total=12)
        self.poll()
        row = self.rows()['fast4']
        self.assertEqual(row['fraction'], .25)
        self.assertEqual(row['detail'], 'Verifying files')
        self.assertTrue(row['can_cancel'])
        self.assertFalse(self.manager.busy)  # A web-owned download must not hold launcher close or update.

        def stop(mode, job_id):
            self.values[mode]['install'] = dict(state='cancelled', job_id=job_id)
        self.backend.cancel.side_effect = stop
        self.manager.request('cancel', 'fast4', row['job_id'])
        self.poll()
        self.backend.cancel.assert_called_once_with('fast4', job_id='job')
        row = self.rows()['fast4']
        self.assertEqual((row['detail'], row['action']), ('Paused', 'Continue'))
        self.assertTrue(row['show_download'])

    def test_retry_preserves_the_reason_with_a_partial(self):
        self.values['fast4'].update(resume_bytes=400, install=dict(state='failed', reason='disk_space'))
        self.poll()
        row = self.rows()['fast4']
        self.assertEqual((row['detail'], row['action'], row['bytes']), ('Not enough disk space', 'Retry', 600))
        self.values['fast4'].update(resume_bytes=0, install=dict(state='failed', reason='interrupted'))
        self.assertEqual(self.rows()['fast4']['action'], 'Continue')

    def test_start_is_asynchronous_and_update_pauses_only_its_worker(self):
        self.poll()
        entered, release = Event(), Event()

        def start(_mode):
            entered.set()
            release.wait(2)

        self.backend.start.side_effect = start
        self.manager.request('install', 'fast4', None)
        try:
            self.manager.poll(self.engine)
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.manager.busy)
            self.manager.pause_owned()
        finally:
            release.set()
            self.manager.thread.join(2)
        self.backend.start.assert_called_once_with('fast4')
        self.backend.cancel.assert_not_called()
        self.backend.cancel_owned.assert_called()


if __name__ == '__main__':
    unittest.main()
