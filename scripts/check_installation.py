"""Check an installation source bundle without models, a GPU or network access."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'triton', 'transformers', 'diffusers'):
            raise RuntimeError('Installation unexpectedly imports ' + fullname)


def main():
    sys.meta_path.insert(0, NoModelImports())
    from freevideo_engine import (cli, comfy_launcher, comfy_launcher_runtime, hf_image_catalog,
                                  hf_image_models, hf_video_catalog, hf_video_models, local_image_catalog,
                                  local_model_copy, local_model_picker, local_video_catalog, modern_launcher)
    from freevideo_engine.desktop_runtime import check_launcher_payload, materialize_source, source_files

    spec = importlib.util.spec_from_file_location('freevideo_install_check', ROOT / '__init__.py',
                                                submodule_search_locations=[str(ROOT)])
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    assert callable(entry.comfy_entrypoint)
    assert (ROOT / entry.WEB_DIRECTORY).is_dir()
    assert callable(cli.main) and callable(comfy_launcher.main)
    assert callable(modern_launcher.main)
    assert callable(hf_video_catalog.inspect) and callable(hf_video_models.VideoModels)
    assert callable(hf_image_catalog.inspect) and callable(hf_image_models.ImageModels)
    assert callable(local_image_catalog.inspect) and callable(local_video_catalog.inspect)
    assert callable(local_model_picker.select) and callable(local_model_copy.copy_files)
    assert (ROOT / 'freevideo_engine/launcher/Main.qml').is_file()

    workflow = json.loads((ROOT / 'example_workflows' / comfy_launcher_runtime.TEMPLATE).read_text(encoding='utf-8'))
    assert {node['type'] for node in workflow['nodes']} == {'FreeVideoMedia', 'FreeVideoGenerate'}
    with tempfile.TemporaryDirectory(prefix='FreeVideo install 中文 ') as folder:
        root = Path(folder)
        source = materialize_source(ROOT, root / 'launcher')
        payload = check_launcher_payload(source, root)
        assert all((source / path.relative_to(ROOT)).read_bytes() == path.read_bytes()
                   for path in source_files(ROOT))
        for name in ('freevideo_engine/hf_video_catalog.py', 'freevideo_engine/hf_video_models.py',
                     'freevideo_engine/hf_image_catalog.py', 'freevideo_engine/hf_image_models.py',
                     'freevideo_engine/image_gen/qwen_checkpoint.py',
                     'freevideo_engine/image_gen/variant_smoke.py', 'freevideo_engine/hf_model_jobs.py',
                     'freevideo_engine/local_image_catalog.py', 'freevideo_engine/local_video_catalog.py',
                     'freevideo_engine/local_model_copy.py', 'freevideo_engine/local_model_picker.py',
                     'web/settings_video_models.js'):
            assert (source / name).is_file(), 'Missing model selector resource: ' + name
        assert not any((source / name).exists() for name in ('benchmarks', 'docs', 'tests', 'AGENTS.md'))
        for name in ('rewriter.py', 'rewriter_setup.py', 'rewriter_models.json'):
            assert not (source / 'freevideo_engine' / name).exists(), 'Retired feature in package: ' + name
        assert not (source / 'web/rewriter.js').exists(), 'Retired prompt model panel in package'
        from freevideo_engine import support_report
        report = dict(success=False, prompt='PRIVATE PROMPT',
                      error_message='PRIVATE PROMPT', token='PRIVATE TOKEN',
                      resources=dict(ram=dict(process_tree_peak_guard_bytes=123)))
        output = root / 'video.mp4'
        report_file = support_report.write(output, report)
        assert report_file and report_file.name == 'video.debug.json'
        saved = report_file.read_text(encoding='utf-8')
        assert 'PRIVATE' not in saved
        assert json.loads(saved)['memory']['video']['ram']['process_tree_peak_guard_bytes'] == 123
    print(json.dumps(dict(success=True, resources=len(payload['resources']),
                          checks=['CLI and launcher imports', 'ComfyUI entry and workflow',
                                  'durable launcher source', 'packaged installation manifests']), indent=2))


if __name__ == '__main__':
    main()
