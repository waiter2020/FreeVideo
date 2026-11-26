import sys
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from freevideo_engine.comfy_h3_smoke import _cancel, _wait_for_output


class SmokeQueueTests(unittest.TestCase):
    def test_pending_cancel_removes_only_the_smoke_prompt(self):
        pending = [(0, 'video'), (1, 'smoke'), (2, 'image')]
        interrupt = Mock()

        def delete(matches):
            before = len(pending)
            pending[:] = [row for row in pending if not matches(row)]
            return len(pending) != before

        queue = SimpleNamespace(mutex=RLock(), delete_queue_item=delete)
        with patch.dict(sys.modules, {'nodes': SimpleNamespace(interrupt_processing=interrupt)}):
            _cancel(queue, 'smoke')
        self.assertEqual(pending, [(0, 'video'), (2, 'image')])
        interrupt.assert_not_called()

    def test_running_cancel_waits_for_terminal_state(self):
        queue = SimpleNamespace(mutex=RLock(), delete_queue_item=lambda _: False,
                                get_current_queue=lambda: ([(0, 'smoke')], []))
        interrupt = Mock()
        port = Mock()
        port.history.side_effect = [None, {'status': {'status_str': 'error'}}]
        port.prompt_state.return_value = 'running'
        check = Mock(side_effect=InterruptedError('cancelled'))
        with patch.dict(sys.modules, {'nodes': SimpleNamespace(interrupt_processing=interrupt)}), \
                patch('freevideo_engine.comfy_h3_smoke.time.sleep'):
            with self.assertRaises(InterruptedError):
                _wait_for_output(port, queue, 'smoke', check)
        self.assertEqual(port.history.call_count, 2)
        interrupt.assert_called_once_with()

    def test_success_and_failure_use_the_test_prompts_history(self):
        outputs = {'92': {'images': [{'filename': 'video.mp4'}]}}
        port = Mock()
        port.history.return_value = {'status': {'status_str': 'success'}, 'outputs': outputs}
        self.assertIs(_wait_for_output(port, None, 'smoke', lambda: None), outputs)
        port.history.return_value['status']['status_str'] = 'error'
        port.generation_error.return_value = {'message': 'checkpoint error'}
        with self.assertRaisesRegex(ValueError, 'checkpoint error'):
            _wait_for_output(port, None, 'smoke', lambda: None)
