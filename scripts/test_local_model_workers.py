"""Check commands issued from a ComfyUI custom-node namespace."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class LocalModelWorkersTest(unittest.TestCase):
    def setUp(self):
        namespace = ModuleType('freevideo_custom_node_test')
        self.namespace = namespace.__name__
        self.source = Path(__file__).resolve().parents[1]
        namespace.__path__ = [str(self.source)]
        sys.modules[self.namespace] = namespace
        self.addCleanup(self._remove_namespace)

    def _remove_namespace(self):
        for name in tuple(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + '.'):
                del sys.modules[name]

    def _plugin_module(self, name):
        return importlib.import_module(self.namespace + '.freevideo_engine.' + name)

    def test_picker_starts_the_standalone_module_from_a_custom_node(self):
        picker = self._plugin_module('local_model_picker')
        selected = str(self.source / 'chosen-model.safetensors')
        child = Mock(returncode=0)
        child.communicate.return_value = (json.dumps(selected).encode(), b'')
        with patch.object(picker.processes, 'popen', return_value=child) as start:
            self.assertEqual(picker.select('engine-python', self.source, 'image'), selected)
        command = start.call_args.args[0]
        self.assertEqual(command[0], 'engine-python')
        self.assertEqual(command[-2:], ['freevideo_engine.local_model_picker', 'image'])

    def test_video_validation_starts_the_standalone_module_from_a_custom_node(self):
        videos = self._plugin_module('hf_video_models')
        managed = self._plugin_module('managed')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = videos.VideoModels(root)

            def child(machine, folder, name, args, **_):
                if name == 'smoke':
                    output = Path(args[args.index('--out') + 1])
                    output.write_bytes(b'video fixture' * 5)
                    output.with_suffix('.request.json').write_text('{"success":true}', encoding='utf-8')
                else:
                    self.assertEqual(name, 'smoke-validation')
                    self.assertEqual(args[:3], ['-m', 'freevideo_engine.hf_video_models', 'validate-video'])

            with (patch.object(managed, 'command_for', side_effect=lambda machine, command, args: ['engine-python', *args]),
                  patch.object(manager, '_child', side_effect=child) as calls):
                manager._smoke({}, root, root / 'cache', None)
            self.assertEqual(calls.call_count, 2)


if __name__ == '__main__':
    unittest.main()
