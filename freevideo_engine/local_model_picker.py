"""Run the native Windows picker in its own managed Python process."""
import json
import logging
import os
import subprocess
import sys

from . import processes


FILE_TYPES = {
    'image': ('Image denoiser / 图像去噪权重', ('*.safetensors', '*.gguf')),
    'moss': ('Image denoiser or shard index / 图像去噪权重或分片索引',
             ('*.safetensors', '*.safetensors.index.json', '*.gguf')),
    'text_encoder': ('Encoder weights / 编码器权重', ('*.safetensors',)),
    'encoder_shards': ('Encoder weights or shard index / 编码器权重或分片索引',
                       ('*.safetensors', '*.safetensors.index.json')),
    'vae': ('VAE weights / VAE 权重', ('*.safetensors',)),
    'cache': ('FreeVideo cache manifest / FreeVideo 缓存清单', 'manifest.json'),
    'model-file': ('Model weights / 模型权重', ('*.safetensors', '*.pt', '*.pth', '*.onnx', '*.json', '*.gguf')),
}


def select(python, source, kind):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    # ComfyUI imports the custom node under a process-local alias. A new Python
    # cannot resolve that alias and needs the package's standalone module name.
    command = processes.module_command('freevideo_engine.local_model_picker', kind)
    command[0] = python
    process = processes.popen(command, env=env, cwd=source,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                              supervise=True)
    try:
        stdout, stderr = process.communicate(timeout=10 * 60)
        if not process.returncode:
            value = json.loads(stdout.decode('utf-8'))
            if value is None or isinstance(value, str) and os.path.isabs(value):
                return value
            raise ValueError('Invalid picker response')
        logging.getLogger(__name__).warning('Local model picker exited with code %s: %s', process.returncode,
                                           stderr.decode('utf-8', 'replace').strip()[-1200:])
    except ValueError:
        logging.getLogger(__name__).warning('Local model picker returned an invalid response')
    except subprocess.TimeoutExpired:
        logging.getLogger(__name__).warning('Local model picker timed out')
    finally:
        if process.poll() is None:
            processes.stop(process)
    raise ValueError('Could not open the local file picker. Open FreeVideo on the Windows desktop and try again.')


def main(kind):
    from tkinter import Label, Tk, filedialog

    root = Tk()
    try:
        root.title('FreeVideo')
        root.resizable(False, False)
        Label(root, text='Select local model / 选择本地模型').pack(padx=28, pady=16)
        root.attributes('-topmost', True)
        root.update()
        if kind in ('source', 'comfy_h3', 'model-directory'):
            title = ('Select ComfyUI MiniMax H3 models / 选择 ComfyUI MiniMax H3 权重目录'
                     if kind == 'comfy_h3' else 'Select VDN-H3 source folder / 选择 VDN-H3 原始权重目录')
            if kind == 'model-directory':
                title = 'Select model directory / 选择模型目录'
            value = filedialog.askdirectory(parent=root, title=title, mustexist=True)
        else:
            title = 'Select model file / 选择模型文件'
            value = filedialog.askopenfilename(parent=root, title=title, filetypes=(FILE_TYPES[kind],))
    finally:
        root.destroy()
    print(json.dumps(value or None), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
