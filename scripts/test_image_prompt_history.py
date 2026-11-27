"""Prompt history follows the displayed text and the works that still use it."""
import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from freevideo_engine import private_refs, private_store, work_thumbnails
from freevideo_engine.image_gen import service


class ImagePromptHistoryTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix='fv-image-history-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_root = self.root / 'input'
        self.input_root.mkdir()
        self.output_root = self.root / 'output'
        self.output_root.mkdir()
        folders = SimpleNamespace(get_input_directory=lambda: str(self.input_root),
                                  get_output_directory=lambda: str(self.output_root))
        self.enterContext(patch.dict('sys.modules', folder_paths=folders))
        self.enterContext(patch.object(private_store, 'store_root', return_value=self.root / 'private'))
        self.enterContext(patch.object(work_thumbnails, 'release'))
        self.service = service.Service(None, folders.get_input_directory, video_busy=lambda: False, folders=folders)
        self.addCleanup(atexit.unregister, self.service.close)
        self.addCleanup(self.service.close)

    def record(self, number, **value):
        path = self.service.output_directory() / f'image-20261001-123456-{number:02d}.jpg'
        path.touch()
        self.service.note_record(path, self.service.work_record(value, 'generate'))
        return path

    def test_forget_the_displayed_prompt_after_the_last_work_is_deleted(self):
        history = 'Put a hat on the cat in @Image1'
        normalized = 'Put a hat on the cat in <Picture 1>'
        first = self.record(1, prompt=normalized, history_prompt=history)
        second = self.record(2, prompt=normalized + ' with sketch instructions', history_prompt=history)
        self.service.history_delete('freevideo-image/' + first.name)
        self.assertEqual(private_refs.forgotten_prompts([history], self.output_root), [])
        self.service.history_delete_many(['freevideo-image/' + second.name])
        self.assertEqual(private_refs.forgotten_prompts([history, normalized], self.output_root), [0])

    def test_a_regenerated_or_undoable_work_overrides_an_old_deletion(self):
        prompt = 'A cat wearing a hat'
        deleted = self.record(1, prompt=prompt)
        self.service.history_delete('freevideo-image/' + deleted.name)
        self.assertEqual(private_refs.forgotten_prompts([prompt], self.output_root), [0])
        replacement = self.record(2, prompt=prompt)
        self.assertEqual(private_store.forgotten([prompt]), [0])
        self.assertEqual(private_refs.forgotten_prompts([prompt], self.output_root), [])
        with patch.object(service.threading, 'Timer'):
            answer = self.service.history_trash(['freevideo-image/' + replacement.name])
        self.assertIsNotNone(answer['batch'])
        self.assertFalse(replacement.exists())
        self.assertEqual(private_refs.forgotten_prompts([prompt], self.output_root), [])
        self.service.finish_trash(everything=True)
        self.assertEqual(private_refs.forgotten_prompts([prompt], self.output_root), [0])

    def test_history_prompt_must_be_nonempty_text_of_supported_length(self):
        for invalid in (None, 12, '  ', 'x' * (service.PROMPT_LIMIT + 1)):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, '^invalid_prompt$'):
                    self.service.work_record({'prompt': 'valid', 'history_prompt': invalid}, 'generate')

    def upload(self):
        name = 'freevideo-image-ref/01234567-89ab-4cde-8f01-23456789abcd.png'
        path = self.input_root / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
        return name, path

    def test_opted_in_busy_retry_keeps_the_same_upload_and_only_queues_generation_fields(self):
        name, path = self.upload()
        request = dict(prompt='A cat in <Picture 1>', width=256, height=256, images=[name],
                       history_prompt='A cat in @Image1', retry_busy=True)
        self.service.segmenting = True
        with self.assertRaisesRegex(ValueError, '^busy$'):
            self.service.start('generate', request)
        self.assertTrue(path.is_file())
        self.service.segmenting = False
        self.service._busy = True
        self.service.queue_thread = Mock(**{'is_alive.return_value': True})

        def prepare(action, value, context, *, waiting):
            self.assertNotIn('history_prompt', value)
            self.assertNotIn('retry_busy', value)
            prepared = self.service.request(value)
            self.assertEqual(prepared['images'], [str(path)])
            self.assertEqual(prepared['prompt'], request['prompt'])
            self.assertEqual(context.work_record['prompt_hash'], private_store.name('prompt', request['history_prompt']))
            return self.root, {}, prepared

        with patch.object(self.service, 'prepare', side_effect=prepare), patch.object(service, 'Context', wraps=service.Context) as context:
            self.assertIn('job', self.service.start('generate', request))
            self.assertNotIn('history_prompt', context.call_args.args[1])
            self.assertNotIn('retry_busy', self.service.pending[0]['value'])

    def test_other_rejections_still_remove_temporary_uploads(self):
        name, path = self.upload()
        self.service.segmenting = True
        with self.assertRaisesRegex(ValueError, '^busy$'):
            self.service.start('generate', dict(prompt='A cat', images=[name]))
        self.assertFalse(path.exists())

        name, path = self.upload()
        with self.assertRaisesRegex(ValueError, '^invalid_prompt$'):
            self.service.start('generate', dict(prompt='A cat', images=[name], history_prompt='', retry_busy=True))
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
