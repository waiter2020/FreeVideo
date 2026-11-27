"""Model slots for prompt processing, visual tools, and live previews."""
import hashlib
import json
from pathlib import Path
import re

from .model_components import Slot


PROMPT_REQUIRED = {'config.json', 'generation_config.json', 'preprocessor_config.json',
                   'tokenizer_config.json', 'video_preprocessor_config.json'}
PROMPT_SUPPORT = {'chat_template.json', 'chat_template.jinja', 'merges.txt', 'tokenizer.json',
                  'vocab.json', 'added_tokens.json', 'special_tokens_map.json'}
PROMPT_SHARD = re.compile(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors')
PROMPT_INDEX = 'model.safetensors.index.json'
PROMPT_SINGLE = 'model.safetensors'


def prompt_layout(names):
    missing = PROMPT_REQUIRED - names
    if missing:
        raise ValueError('Missing prompt model files: ' + ', '.join(sorted(missing)))
    if 'tokenizer.json' not in names and not {'merges.txt', 'vocab.json'} <= names:
        raise ValueError('Choose a prompt model bundle that includes its tokenizer.')
    shards = {name for name in names if PROMPT_SHARD.fullmatch(name)}
    if PROMPT_INDEX in names:
        if not shards or PROMPT_SINGLE in names:
            raise ValueError('Provide one complete set of prompt model shards and their index.')
        weights = shards | {PROMPT_INDEX}
    elif PROMPT_SINGLE in names and not shards:
        weights = {PROMPT_SINGLE}
    else:
        raise ValueError('Provide a prompt model safetensors file or its complete shard set.')
    return PROMPT_REQUIRED | (PROMPT_SUPPORT & names) | weights


def validate_prompt(folder):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen3_vl' or config.get_text_config().hidden_act != 'silu':
        raise ValueError('This checkpoint needs a different runtime. Choose a dense Qwen3-VL Instruct model.')
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if not all(getattr(processor, name, None) is not None for name in ('tokenizer', 'image_processor', 'video_processor')):
        raise ValueError('The prompt model must support text, picture, and video inputs.')
    prompt = processor.apply_chat_template([dict(role='user', content=[dict(type='text', text='Describe the scene.')])],
                                           tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str) or 'Describe the scene.' not in prompt:
        raise ValueError('The prompt model needs a compatible chat template.')
    GenerationConfig.from_pretrained(folder, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation='sdpa')
    expected = {name: tuple(parameter.shape) for name, parameter in model.named_parameters(remove_duplicate=False)}
    tied = config.tie_word_embeddings or config.get_text_config().tie_word_embeddings
    seen = {}
    actual = {}
    for path in folder.glob('model*.safetensors'):
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                entry = reader.get_slice(key)
                if key in seen or entry.get_dtype() not in ('BF16', 'F16', 'F32'):
                    raise ValueError('The prompt checkpoint contains duplicate or unsupported tensor data.')
                seen[key] = path.name
                actual[key] = tuple(entry.get_shape())
    if (folder / PROMPT_INDEX).is_file():
        index = json.loads((folder / PROMPT_INDEX).read_text(encoding='utf-8'))['weight_map']
        if seen != index:
            raise ValueError('The prompt weight index disagrees with its shards.')
    if tied and 'lm_head.weight' not in actual:
        expected.pop('lm_head.weight', None)
    if actual != expected:
        raise ValueError('The prompt model tensors do not match its Qwen3-VL configuration.')


def _progress(callback):
    def report(row):
        done, total = row.get('done'), row.get('total')
        if isinstance(done, (int, float)) and isinstance(total, (int, float)):
            callback(done, total)
    return report


def _ensure(root, loader):
    def install(progress, check):
        report = _progress(progress)
        loader(root, lambda **row: report(row), check)
    return install


def _verify_pin(path, row):
    try:
        if path.stat().st_size == row['bytes']:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']:
                    return
    except OSError:
        pass
    raise ValueError(f'The built-in {path.name} file is missing or damaged.')


def _verify_pack(folder, rows):
    for row in rows:
        _verify_pin(folder / row['file'], row)


def _install_prompt(root, progress, check):
    from .locking import runtime_lock
    from .prompt_vlm import assets
    with runtime_lock(Path(root) / 'prompt-vlm-install.lock', inherit=False):
        assets.install(root, _progress(progress), check)


def _single(slot, label, label_zh, consumers, path, format, validate, **options):
    return Slot(id=slot, group='visual', label=label, label_zh=label_zh, consumers=consumers,
                files={path.name: path}, format=format, validate=lambda folder: validate(folder / path.name),
                **options)


def slots(root, machine):
    from . import depth_reference, frame_interp, frame_preview, frame_preview_decoder, frame_upscale, two_pass
    from .image_gen import back_view, face_box, image_pack, segment, step_preview
    from .prompt_vlm import assets

    model_root = machine.get('model_root') or Path(root) / 'models'
    prompt = assets.catalog()
    prompt_path = assets.directory(root)
    depth, rife, upscale = depth_reference.catalog(), frame_interp.catalog(), frame_upscale.catalog()
    seg = image_pack.rows(group='segment')[0]
    depth_path, rife_path, upscale_path = (depth_reference.default_weight(root), frame_interp.default_weight(root),
                                           frame_upscale.default_weight(root))
    latent_path = two_pass.default_checkpoint(root, model_root)
    segment_path = segment.checkpoint_path(model_root)
    return [
        Slot(id='prompt-vlm', group='prompt', label='Prompt and vision model', label_zh='提示词与视觉模型',
             consumers=('Prompt enhancement', 'Visual questions', 'Shot analysis', 'Director'),
             files={row['file']: prompt_path / row['file'] for row in prompt['files']},
             format='Dense Qwen3-VL Instruct, Transformers Safetensors', validate=validate_prompt,
             layout=prompt_layout, source=prompt['repo'], license=prompt['license'],
             install_default=lambda progress, check: _install_prompt(root, progress, check),
             default_status=lambda: dict(installed=assets.default_ready(root), label=prompt['repo'].rsplit('/', 1)[-1]),
             verify_default=lambda: _verify_pack(prompt_path, prompt['files']),
             default_bytes=sum(row['bytes'] for row in prompt['files'])),
        _single('depth-reference', 'Depth reference', '深度参考', ('Depth video reference',),
                depth_path, 'Depth Anything V2 Small .pth',
                lambda path: depth_reference._load_model(path, 'cpu'), source=depth['repo'], license=depth['license'],
                verify_default=lambda: _verify_pin(depth_path, depth['files'][0]),
                install_default=_ensure(root, depth_reference.ensure_weights), default_bytes=depth['files'][0]['bytes']),
        _single('frame-interpolation', 'Frame interpolation', '视频补帧', ('High frame rate export',),
                rife_path, 'RIFE v4.25 Safetensors',
                lambda path: frame_interp.model_between(path, 'cpu'), source=rife['repo'], license=rife['license'],
                verify_default=lambda: _verify_pin(rife_path, rife['files'][0]),
                install_default=_ensure(root, frame_interp.ensure_weights), default_bytes=rife['files'][0]['bytes']),
        _single('frame-upscale', '4K frame upscale', '视频帧 4K 放大', ('4K video upscale',),
                upscale_path, 'Real-ESRGAN General x4v3 .pth',
                lambda path: frame_upscale.load(path, 'cpu'), source=upscale['repo'], license=upscale['license'],
                verify_default=lambda: _verify_pin(upscale_path, upscale['files'][0]),
                install_default=_ensure(root, frame_upscale.ensure_weights), default_bytes=upscale['files'][0]['bytes']),
        _single('h3-latent-upscaler', 'H3 latent upscaler', 'H3 潜空间放大', ('Two-pass video', 'Video upscale'),
                latent_path, 'H3 3D v1 Safetensors', validate_upscaler,
                source=two_pass.UPSCALER['repo'], license='MIT',
                verify_default=lambda: _verify_pin(latent_path, two_pass.UPSCALER),
                install_default=lambda progress, check: two_pass.ensure_checkpoint(root=root, model_directory=model_root,
                                                                                   progress=progress, check=check,
                                                                                   use_selected=False),
                default_bytes=two_pass.UPSCALER['bytes']),
        _single('sam2', 'Interactive segmentation', '交互分割', ('Picture selection', 'Region editing'),
                segment_path, 'SAM 2.1 Hiera Small .pt', segment.build_model,
                source=seg['repo'], license='Apache-2.0', default_bytes=seg['bytes'],
                verify_default=lambda: _verify_pin(segment_path, seg),
                install_default=lambda progress, check: image_pack.install(model_root, _progress(progress), check,
                                                                            engine_root=root, group='segment')),
        Slot(id='face-box', group='visual', label='Face detector', label_zh='人脸检测',
             consumers=('Face crop',), files={face_box.CHECKPOINT: face_box.WEIGHTS}, format='YuNet .pth',
             validate=lambda folder: face_box.load(folder / face_box.CHECKPOINT), verify_default=face_box.weights_bytes,
             source='kornia/yunet', license='MIT'),
        _single('pose-detector', 'Pose detector', '姿态检测', ('Character back view',),
                back_view.default_checkpoint(image_pack.base(model_root)), 'Keypoint R-CNN ResNet50 FPN .pth',
                back_view.pose_model, source='torchvision'),
        _single('video-preview-linear', 'Video preview projection', '视频预览投影', ('Live video preview',),
                frame_preview.WEIGHTS, 'H3 24-channel affine map JSON', frame_preview.load_weights, source='FreeVideo'),
        _single('video-preview-decoder', 'Video preview decoder', '视频预览解码器', ('Live video preview',),
                frame_preview_decoder.WEIGHTS, 'H3 preview decoder Safetensors',
                lambda path: frame_preview_decoder.load('cpu', path), source='FreeVideo'),
        _single('h3-image-preview', 'H3 Image preview projection', 'H3 图像预览投影', ('Live H3 Image preview',),
                step_preview.WEIGHTS, 'H3 Image 24-channel affine map JSON', step_preview.load_weights, source='FreeVideo'),
    ]


def validate_upscaler(path):
    import torch
    from safetensors import safe_open
    from .latent_upscale import LatentResizer3D

    with torch.device('meta'):
        model = LatentResizer3D()
    expected = {name: tuple(parameter.shape) for name, parameter in model.state_dict().items()}
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        shapes = {}
        for key in reader.keys():
            name = key.removeprefix('upscaler.')
            item = reader.get_slice(key)
            if name in shapes or item.get_dtype() not in ('BF16', 'F16', 'F32'):
                raise ValueError('Invalid latent upscaler tensor data.')
            shapes[name] = tuple(item.get_shape())
    if shapes != expected:
        raise ValueError('Expected H3 latent upscaler v1 weights.')
