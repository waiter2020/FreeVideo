"""Run managed MiniMax H3 base presets in the existing native ComfyUI queue."""
from contextlib import contextmanager
from functools import cache
import json
import logging
import math
import mmap
import os
from pathlib import Path
from threading import RLock
import time
import uuid
import warnings


PRESET = 'native_h3_preset'
VIDEO = 'native_h3_video'
ERROR = 'native_h3_error'
SMOKE_OUTPUT_NODE = '92'
LOADER = 'FreeVideoH3Checkpoint'
CONDITIONING = 'FreeVideoH3Conditioning'
RELEASE_SAMPLER = 'FreeVideoH3ReleaseSampler'
_ROLES = {'model': ('model', 'checkpoint'), 'text_encoder': ('text_encoder', 'encoder'),
          'video_vae': ('video_vae',), 'audio_vae': ('audio_vae',)}
_TYPES = {'fl2va', 'ref2va'}
_REFERENCE_OPTIONS = ('early_steps', 'guide', 'voice_only', 'keep_source_audio', 'depth', 'lines', 'range', 'segment')
_LOOKUP = None
_SMOKE = {}
_LEASES = {}
_ACTIVE = {}
_LOCK = RLock()
_LOG = logging.getLogger(__name__)


def _problem(en, zh):
    return ValueError(f'{en} / {zh}')


def _variant(profile):
    value = profile.get('preset', profile.get('variant'))
    if value not in _TYPES:
        raise _problem('The selected H3 checkpoint type is unsupported.', '不支持所选 H3 模型类型。')
    return value


def _files(profile):
    assets = profile.get('assets') or profile
    files = {}
    for key, aliases in _ROLES.items():
        path = next((assets[name] for name in aliases if assets.get(name)), None)
        if not isinstance(path, (str, Path)):
            raise _problem('The selected H3 preset is missing a model component.', '所选 H3 模型缺少组件。')
        path = Path(path)
        if not path.is_absolute() or path.suffix.lower() != '.safetensors' or not path.is_file():
            raise _problem('A file in the selected H3 preset is missing or invalid.', '所选 H3 模型的文件不存在或格式无效。')
        files[key] = str(path)
    return files


def _profile(identity, encoder_component=None):
    with _LOCK:
        profile = _SMOKE.get(identity)
    if profile is None and _LOOKUP is not None:
        profile = (_LOOKUP(identity, encoder_component=encoder_component) if encoder_component is not None
                   else _LOOKUP(identity))
    if not isinstance(profile, dict) or profile.get('id') != identity:
        raise _problem('The selected H3 preset is no longer installed.', '所选 H3 模型已不可用，请重新选择。')
    return profile


@contextmanager
def allow_smoke(profile):
    """Allow a verified installation candidate to load while its smoke prompt runs."""
    identity = profile['id']
    _variant(profile)
    _files(profile)
    with _LOCK:
        if identity in _SMOKE:
            raise RuntimeError('A smoke test is already active for this H3 model.')
        _SMOKE[identity] = profile
    try:
        yield identity
    finally:
        with _LOCK:
            _SMOKE.pop(identity, None)


def _frame_count(seconds, frames=0):
    if frames:
        if type(frames) is int and 39 <= frames <= 362 and (frames - 5) % 17 == 0:
            return frames
        raise _problem('Set Total frames to 39–362 in steps of 17 for MiniMax H3 Base.',
                       'MiniMax H3 底模的总帧数须为 39 至 362，间隔 17 帧。')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 39 / 24 <= seconds <= 362 / 24:
        raise _problem('Choose a duration from 1.625 to about 15 seconds for MiniMax H3 Base.',
                       'MiniMax H3 底模的时长须为 1.625 秒至约 15 秒。')
    frames = math.ceil(seconds * 24)
    return frames + (5 - frames) % 17


def _unsupported(inputs):
    if any(inputs.get(name) for name in ('preview', 'refine_from', 'upscale_from', 'upscale_to',
                                          'continue_from', 'first_frame_reference', 'voice_reference', 'hide_tail',
                                          'conditioning', 'loras', 'two_pass')) or inputs.get('step_cache', 'none') not in ('', 'none'):
        raise _problem('Clear LoRAs, continuations, previews and other VDN-specific options, or switch to the FreeVideo model in Settings.',
                       '请关闭 LoRA、续接、预览和其他 VDN 专用选项，或在设置里切回 FreeVideo 模型。')
    if inputs.get('upscale_factor') not in (None, '', '2', 2):
        raise _problem('Use the default upscale factor with single-pass MiniMax H3 Base.',
                       'MiniMax H3 底模为单次采样，请将放大倍率设为默认值。')


def _check_media(variant, anchors, references):
    if variant == 'fl2va' and references:
        raise _problem('Use a Ref2VA preset for references. FL2VA takes first and last frames.',
                       '参考素材请选 Ref2VA；FL2VA 支持首尾帧。')
    if variant == 'ref2va' and anchors:
        raise _problem('Use an FL2VA preset to set first or last frames.', '需要设置首尾帧时，请选择 FL2VA 模型。')


def _check_reference(row):
    if any(row.get(key) for key in _REFERENCE_OPTIONS) or row.get('use') == 'original':
        raise _problem('A selected reference uses an option unavailable in native H3.', '所选参考素材包含 MiniMax H3 底模不支持的选项。')


def _preflight_media(variant, inputs, graph):
    anchors, references = any(inputs.get(name) for name in ('first', 'last')), bool(inputs.get('references'))
    link = inputs.get('media')
    if link and (anchors or references):
        raise _problem('Use the Media panel or direct media sockets for native H3.', '请选择媒体面板或直连媒体端口，避免重复。')
    source = graph.get(link[0]) if isinstance(link, list) and len(link) == 2 and isinstance(link[0], str) else None
    if isinstance(source, dict) and source.get('class_type') == 'FreeVideoMedia' and isinstance(source.get('inputs'), dict):
        from .comfy_assets import REFERENCE_INPUTS

        media = source['inputs']
        anchors = any(media.get(name) for name in ('first', 'last'))
        references = any(media.get(name) for name in (*REFERENCE_INPUTS, 'reference_audio'))
        if media.get('audio_use') == 'original':
            raise _problem('Native H3 presets do not support original audio passthrough.', 'MiniMax H3 底模不支持原样使用音频。')
        raw = media.get('assets')
        if isinstance(raw, str) and len(raw) <= 128 * 1024:
            try:
                rows = json.loads(raw)
            except ValueError:
                rows = None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict) or row.get('enabled') is False:
                        continue
                    role = row.get('role')
                    if role == 'middle':
                        raise _problem('Native H3 presets do not support middle frames.', 'MiniMax H3 底模不支持中间帧。')
                    anchors |= role in ('first', 'last')
                    references |= role == 'reference'
                    if role == 'reference':
                        _check_reference(row)
    _check_media(variant, anchors, references)


def _graph(profile_id, variant, inputs, prefix, *, frames, steps, owner=None, smoke=False, encoder_component=None):
    nodes = {}

    def add(name, class_type, **values):
        identity = prefix + name
        nodes[identity] = {'class_type': class_type, 'inputs': values}
        return identity

    loader_inputs = {'preset': profile_id}
    if encoder_component is not None:
        loader_inputs['encoder_component'] = encoder_component
    if owner is not None:
        loader_inputs['owner'] = owner
    assets = add('models', LOADER, **loader_inputs)
    media = {key: inputs[key] for key in ('first', 'last', 'references', 'media') if inputs.get(key) is not None}
    if smoke:
        image = add('smoke_image', 'EmptyImage', width=64, height=64, batch_size=1, color=8421504)
        media['smoke_image' if variant == 'ref2va' else 'first'] = [image, 0]
        if variant == 'fl2va':
            tail = add('smoke_tail', 'EmptyImage', width=64, height=64, batch_size=1, color=5273312)
            media['last'] = [tail, 0]
    cond = add('condition', CONDITIONING, clip=[assets, 1], vae=[assets, 2], audio_vae=[assets, 3],
               variant=variant, text=inputs['text'], width=inputs['width'], height=inputs['height'], frames=frames, **media)
    guider = add('guider', 'BasicGuider', model=[assets, 0], conditioning=[cond, 0])
    scheduler = add('schedule', 'BasicScheduler', model=[assets, 0], scheduler='simple', steps=steps, denoise=1.)
    sampler = add('sampler', 'KSamplerSelect', sampler_name='res_multistep')
    noise = add('noise', 'RandomNoise', noise_seed=inputs['seed'])
    latent = add('sample', 'SamplerCustomAdvanced', noise=[noise, 0], guider=[guider, 0],
                 sampler=[sampler, 0], sigmas=[scheduler, 0], latent_image=[cond, 1])
    decode = add('release_sampling', RELEASE_SAMPLER, latent=[latent, 0], model=[assets, 0], clip=[assets, 1])
    images = add('images', 'VAEDecode', samples=[decode, 0], vae=[assets, 2])
    audio = add('audio', 'VAEDecodeAudio', samples=[decode, 0], vae=[assets, 3])
    video = add('video', 'CreateVideo', images=[images, 0], audio=[audio, 0], fps=24.)
    return nodes, [video, 0]


def _owned_node(identity, row, owner):
    metadata = row.get('_meta')
    return (isinstance(identity, str) and identity.startswith(f'freevideo_h3_{owner}_')
            and isinstance(metadata, dict) and metadata.get('freevideo_h3_owner') == owner)


def expand_prompt(payload, lookup):
    """Replace only the trusted internals of native preset branches; preserve the owner node."""
    source = payload.get('prompt')
    if not isinstance(source, dict):
        return payload
    nodes = dict(source)
    changed = False
    native_owners, vdn_owners = [], []
    for identity, row in source.items():
        if not isinstance(row, dict) or row.get('class_type') != 'FreeVideoGenerate' or not isinstance(row.get('inputs'), dict):
            continue
        inputs = dict(row['inputs'])
        explicit = PRESET in inputs
        selected = inputs.get(PRESET)
        profile = None
        lookup_error = None
        if not explicit:
            try:
                profile = lookup(None)
                if profile is not None:
                    selected = profile['id']
                    inputs[PRESET] = selected
            except (OSError, TypeError, ValueError, KeyError) as error:
                lookup_error = error
        if selected:
            native_owners.append(identity)
        elif lookup_error is None:
            vdn_owners.append(identity)
        if not selected and lookup_error is None and VIDEO not in inputs and ERROR not in inputs:
            continue
        changed = True
        inputs.pop(VIDEO, None)
        inputs.pop(ERROR, None)
        nodes[identity] = {**row, 'inputs': inputs}
        # A history snapshot may carry an earlier expansion with a different seed.
        for key in tuple(nodes):
            if isinstance(nodes[key], dict) and _owned_node(key, nodes[key], identity):
                del nodes[key]
        if lookup_error is not None:
            inputs[ERROR] = str(lookup_error)
            continue
        if not selected:
            continue
        try:
            if not isinstance(selected, str) or not selected:
                raise _problem('Choose an installed MiniMax H3 Base preset.', '请选择已安装的 MiniMax H3 底模。')
            if profile is None:
                profile = lookup(selected)
            if not isinstance(profile, dict) or profile.get('id') != selected:
                raise _problem('The selected H3 preset is no longer installed.', '所选 H3 模型已不可用，请重新选择。')
            variant = _variant(profile)
            _unsupported(inputs)
            _preflight_media(variant, inputs, source)
            frames = _frame_count(inputs.get('seconds', 10.), inputs.get('frames'))
            steps = inputs.get('base_steps', 20)
            if not isinstance(steps, list) and (type(steps) is not int or not 1 <= steps <= 32):
                raise _problem('Choose 1–32 sampling steps for MiniMax H3 Base.', '采样步数须为 1 至 32。')
            inputs.setdefault('base_steps', steps)
            inputs.setdefault('two_pass', False)
            # Copy only this run's values. Graph links remain ordinary Comfy inputs.
            args = dict(text='', width=1344, height=768, seed=0)
            args.update(inputs)
            from .encoder_checkpoint import BUILTIN_ENCODER
            encoder_component = (profile.get('shared_encoder') or {}).get('id', BUILTIN_ENCODER)
            generated, video = _graph(selected, variant, args, f'freevideo_h3_{identity}_',
                                      frames=frames, steps=steps, owner=identity, encoder_component=encoder_component)
            if any(key in nodes for key in generated):
                raise _problem('Native H3 node IDs conflict with this workflow.', '底模节点与此工作流中的节点编号冲突。')
            for graph_node in generated.values():
                graph_node['_meta'] = {'freevideo_h3_owner': identity}
            nodes.update(generated)
            inputs[VIDEO] = video
        except (OSError, TypeError, ValueError, KeyError) as error:
            inputs[ERROR] = str(error)
    if native_owners and vdn_owners:
        message = str(_problem('Queue native MiniMax H3 Base and VDN FreeVideo outputs as separate prompts.',
                               '请分别提交 MiniMax H3 底模和 VDN 的 FreeVideo 输出。'))
        for identity in (*native_owners, *vdn_owners):
            row = nodes[identity]
            inputs = dict(row['inputs'])
            inputs.pop(VIDEO, None)
            inputs.setdefault(ERROR, message)
            nodes[identity] = {**row, 'inputs': inputs}
        for key, node in tuple(nodes.items()):
            if isinstance(node, dict) and any(_owned_node(key, node, owner) for owner in native_owners):
                del nodes[key]
    return {**payload, 'prompt': nodes} if changed else payload


def smoke_graph(profile_id, *, variant=None, width=256, height=256, frames=39, seed=112358,
                filename_prefix='FreeVideo/.model-smoke/H3'):
    """A deterministic short Comfy graph using the model's exact reusable components."""
    profile = _profile(profile_id) if variant is None else {'preset': variant}
    preset = _variant(profile)
    prompt = '<Picture 1> is the visual reference. A small red balloon floats in a clear blue sky.' if preset == 'ref2va' else 'A small red balloon floats in a clear blue sky.'
    graph, video = _graph(profile_id, preset, dict(text=prompt, width=width, height=height, seed=seed),
                          'freevideo_h3_smoke_', frames=frames, steps=2, smoke=True)
    graph[SMOKE_OUTPUT_NODE] = {'class_type': 'SaveVideo', 'inputs': {'video': video, 'format': 'mp4',
                              'filename_prefix': filename_prefix}}
    return graph


def _runtime_error():
    return _problem('This ComfyUI version lacks the native MiniMax H3 nodes or prompt lifecycle API. Repair or update ComfyUI.',
                    '此 ComfyUI 缺少 MiniMax H3 原生节点或任务生命周期接口，请修复或升级 ComfyUI。')


def _prepare(prompt_id):
    with _LOCK:
        if prompt_id in _LEASES:
            return
    from comfy import model_management as memory
    from server import PromptServer

    from . import comfy_bridge, video_warm
    from .background_compile import COMPILE
    from .encoder_prewarm import IDLE
    from .image_gen.service import release_idle_worker
    from .locking import runtime_lock
    from .prompt_vlm.service import stop_prestart_for_generation
    from .resident_process import OWNER

    COMPILE.stop(final=True)
    video_warm.stop()
    IDLE.stop()
    OWNER.stop_prewarm()
    OWNER.release_idle('native_h3')
    release_idle_worker(PromptServer.instance)
    stop_prestart_for_generation(PromptServer.instance)
    memory.throw_exception_if_processing_interrupted()
    root, _ = comfy_bridge.installation()
    lease = runtime_lock(Path(root) / 'engine.lock', inherit=False, wait_seconds=3)
    try:
        lease.__enter__()
    except BlockingIOError as error:
        raise _problem('Another FreeVideo task still holds the GPU. Wait for it to finish before running native H3.',
                       '显卡仍由另一个 FreeVideo 任务占用，请等它结束后再生成。') from error
    with _LOCK:
        _LEASES[prompt_id] = lease
    memory.throw_exception_if_processing_interrupted()


def _release(prompt_id):
    with _LOCK:
        lease = _LEASES.pop(prompt_id, None)
        _ACTIVE.pop(prompt_id, None)
    if lease is not None:
        try:
            from comfy import model_management as memory
            memory.unload_all_models()
            memory.soft_empty_cache()
        finally:
            lease.__exit__(None, None, None)


def _read_windows_weights(path):
    from comfy.utils import _TYPES
    from safetensors import safe_open
    import torch

    # Validate the safetensors index without reading or mapping the weights.
    with safe_open(path, framework='pt', device='cpu', backend='pread') as reader:
        metadata = reader.metadata()
    # The normal Torch mapping is copy-on-write on Windows and can charge a
    # whole checkpoint against commit. Keep this model's source pages read-only.
    with open(path, 'rb') as source:
        length = int.from_bytes(source.read(8), 'little')
        if not 2 <= length <= 100_000_000:
            raise ValueError('Invalid safetensors header length.')
        header = json.loads(source.read(length))
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
    base = 8 + length
    header.pop('__metadata__', None)
    weights = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The given buffer is not writable')
        for key, entry in header.items():
            dtype = _TYPES[entry['dtype']]
            shape = entry['shape']
            start, end = entry['data_offsets']
            if not 0 <= start <= end <= len(mapping) - base or math.prod(shape) * dtype.itemsize != end - start:
                raise ValueError('Invalid safetensors tensor range: ' + key)
            if start == end:
                weights[key] = torch.empty(shape, dtype=dtype)
            else:
                view = memoryview(mapping)[base + start:base + end]
                weights[key] = torch.frombuffer(view, dtype=dtype).reshape(shape)
    # torch.frombuffer retains each view and the underlying mmap.
    return weights, metadata


def _load_models(preset, encoder_component=None):
    import folder_paths
    from comfy import model_management as memory
    import comfy.sd
    import comfy.utils

    profile = _profile(preset, encoder_component)
    files = _files(profile)
    if os.name == 'nt':
        weights, metadata = _read_windows_weights(files['model'])
        model = comfy.sd.load_diffusion_model_state_dict(weights, metadata=metadata)
        del weights
    else:
        model = comfy.sd.load_diffusion_model(files['model'])
    if model is None:
        raise _problem('ComfyUI could not load this H3 checkpoint.', 'ComfyUI 无法载入所选 H3 底模权重。')
    memory.throw_exception_if_processing_interrupted()
    embedding_directory = folder_paths.get_folder_paths('embeddings')
    if os.name == 'nt':
        weights, metadata = _read_windows_weights(files['text_encoder'])
        weights, _ = comfy.utils.convert_old_quants(weights, model_prefix='', metadata=metadata)
        clip = comfy.sd.load_text_encoder_state_dicts([weights], clip_type=comfy.sd.CLIPType.MINIMAX,
                                                      embedding_directory=embedding_directory)
        del weights
    else:
        clip = comfy.sd.load_clip(ckpt_paths=[files['text_encoder']], clip_type=comfy.sd.CLIPType.MINIMAX,
                                 embedding_directory=embedding_directory)
    encoder = profile.get('shared_encoder')
    if encoder is not None:
        _LOG.info('[FreeVideo] Native H3 %s loaded shared encoder %s (SHA-256 %s) from %s.',
                  _variant(profile), encoder['id'], encoder['sha256'], files['text_encoder'])
    vaes = []
    for key in ('video_vae', 'audio_vae'):
        memory.throw_exception_if_processing_interrupted()
        if os.name == 'nt':
            weights, metadata = _read_windows_weights(files[key])
        else:
            weights, metadata = comfy.utils.load_torch_file(files[key], return_metadata=True)
        vae = comfy.sd.VAE(sd=weights, metadata=metadata)
        vae.throw_exception_if_invalid()
        vaes.append(vae)
    return model, clip, *vaes


def _image(path):
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    with Image.open(path) as source:
        data = np.asarray(ImageOps.exif_transpose(source).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(data).unsqueeze(0)


def _media(first, last, references, media):
    if not media:
        return first, last, list(references or [])
    if first is not None or last is not None or references:
        raise _problem('Use the Media panel or direct media sockets for native H3.', '请选择媒体面板或直连媒体端口，避免重复。')
    if media.get('middle') or media.get('_audio_use') == 'original':
        raise _problem('Native H3 presets do not support middle frames or original audio passthrough.',
                       'MiniMax H3 底模不支持中间帧或原样使用音频。')
    from .comfy_assets import REFERENCE_INPUTS
    connected = media.get('_connected') or {}
    refs = list(media.get('references') or [])
    for key in REFERENCE_INPUTS:
        image = connected.get(key)
        if image is not None:
            refs.extend({'kind': 'image', 'value': image[i:i + 1]} for i in range(image.shape[0]))
    if connected.get('reference_audio') is not None:
        refs.append({'kind': 'audio', 'value': connected['reference_audio']})
    anchors = [connected.get(name) if not media.get(name) else _image(media[name]) for name in ('first', 'last')]
    return *anchors, refs


def _reference_audio(value, path, limit):
    import torch

    if value is None:
        import av

        from .original_audio import _samples, _to_float

        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise _problem('The reference file has no audio track.', '参考文件没有音轨。')
            stream = container.streams.audio[0]
            rate = int(stream.codec_context.sample_rate or 0)
            remaining = int(rate * limit)
            chunks = []
            for frame in container.decode(stream):
                samples = _samples(frame)[..., :remaining]
                if samples.size:
                    chunks.append(torch.from_numpy(_to_float(samples)))
                    remaining -= samples.shape[-1]
                if remaining <= 0:
                    break
        if chunks:
            value = {'waveform': torch.cat(chunks, dim=-1).unsqueeze(0), 'sample_rate': rate}
    waveform, rate = (value.get('waveform'), value.get('sample_rate')) if isinstance(value, dict) else (None, None)
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1
            or waveform.shape[1] not in (1, 2) or type(rate) is not int or not 8000 <= rate <= 192000):
        raise _problem('Connect one mono or stereo audio reference.', '请连接一项单声道或立体声参考音频。')
    cropped = waveform[..., :int(rate * limit)]
    if not cropped.numel() or not torch.isfinite(cropped).all():
        raise _problem('The reference audio has no usable samples.', '参考音频没有可用的采样。')
    return {'waveform': cropped.float(), 'sample_rate': rate}, cropped.shape[-1] / rate


def _reference_video(value, path, limit, *, include_audio):
    import torch
    from comfy_api.latest import InputImpl

    value = value if value is not None else InputImpl.VideoFromFile(str(path))
    if not all(callable(getattr(value, method, None)) for method in ('get_duration', 'get_components')):
        raise _runtime_error()
    seconds = min(limit, float(value.get_duration()))
    if seconds <= 0:
        raise _problem('A reference video is empty.', '参考视频为空。')
    if isinstance(value, InputImpl.VideoFromFile):
        value = value.as_trimmed(start_time=0., duration=seconds, strict_duration=False)
    components = value.get_components()
    frames, rate = components.images, float(components.frame_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise _problem('The reference video has no usable frame rate.', '参考视频没有可用帧率。')
    count = min(math.floor(seconds * 24 + 1e-6), math.floor(frames.shape[0] * 24 / rate + 1e-6))
    if count < 5:
        raise _problem('A native H3 reference video needs at least five frames.', 'MiniMax H3 参考视频至少需要五帧。')
    count -= (count - 5) % 17
    indices = (torch.arange(count, device=frames.device) * rate / 24.).floor().long().clamp_(max=frames.shape[0] - 1)
    seconds = count / 24.
    audio = _reference_audio(components.audio, None, seconds)[0] if include_audio and components.audio else None
    return frames[indices], audio, seconds


def _references(rows, limit):
    groups = {'ref_images': {}, 'ref_videos': {}, 'ref_video_audios': {}, 'ref_audios': {}}
    total = {'video': 0., 'audio': 0.}
    if len(rows) > 12:
        raise _problem('MiniMax H3 Ref2VA accepts up to 12 references.', 'MiniMax H3 Ref2VA 最多接受 12 项参考。')
    for row in rows:
        _check_reference(row)
        kind, value, path = row['kind'], row.get('value'), row.get('path')
        if kind == 'image':
            images = groups['ref_images']
            if len(images) == 9:
                raise _problem('MiniMax H3 Ref2VA accepts up to nine images.', 'MiniMax H3 Ref2VA 最多接受九张图。')
            images[f'ref_image_{len(images)}'] = value if value is not None else _image(path)
        elif kind == 'video':
            videos = groups['ref_videos']
            if len(videos) == 3 or total[kind] >= 15.:
                raise _problem('MiniMax H3 Ref2VA accepts up to three videos totaling 15 seconds.', 'MiniMax H3 Ref2VA 最多接受三个总长 15 秒的视频。')
            images, audio, seconds = _reference_video(value, path, min(limit, 15. - total[kind]),
                                                      include_audio=row.get('sound', True))
            index = len(videos)
            videos[f'ref_video_{index}'] = images
            if audio is not None:
                groups['ref_video_audios'][f'ref_video_audio_{index}'] = audio
            total[kind] += seconds
        elif kind == 'audio':
            audios = groups['ref_audios']
            if len(audios) == 3 or total[kind] >= 15.:
                raise _problem('MiniMax H3 Ref2VA accepts up to three audio files totaling 15 seconds.', 'MiniMax H3 Ref2VA 最多接受三项总长 15 秒的音频。')
            audio, seconds = _reference_audio(value, path, min(limit, 15. - total[kind]))
            audios[f'ref_audio_{len(audios)}'] = audio
            total[kind] += seconds
        else:
            raise _problem('The selected H3 reference type is unsupported.', '不支持所选的 H3 参考类型。')
    return groups


@cache
def runtime_nodes():
    from comfy_api.latest import io

    class Checkpoint(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=LOADER, display_name='FreeVideo H3 preset weights', category='FreeVideo',
                inputs=[io.String.Input('preset'), io.String.Input('owner', optional=True),
                        io.String.Input('encoder_component', optional=True)],
                outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output(), io.Vae.Output()], not_idempotent=True)

        @classmethod
        def fingerprint_inputs(cls, **kwargs):
            # A cached loader would bypass the GPU handoff and execution lease.
            return float('nan')

        @classmethod
        def execute(cls, preset, owner=None, encoder_component=None):
            from .comfy_progress import publish
            from server import PromptServer
            server = PromptServer.instance
            identity = server.last_prompt_id
            if owner and identity:
                with _LOCK:
                    _ACTIVE.setdefault(identity, set()).add(owner)
                publish(server, owner, {'label': 'Preparing video', 'phase': 'starting', 'new_request': True})
            _prepare(identity)
            return io.NodeOutput(*_load_models(preset, encoder_component))

    class Conditioning(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=CONDITIONING, display_name='FreeVideo H3 media', category='FreeVideo',
                inputs=[io.Clip.Input('clip'), io.Vae.Input('vae'), io.Vae.Input('audio_vae'),
                        io.Combo.Input('variant', options=sorted(_TYPES)), io.String.Input('text'),
                        io.Int.Input('width', min=256, max=4096, step=32), io.Int.Input('height', min=256, max=4096, step=32),
                        io.Int.Input('frames', min=39, max=362, step=17), io.Image.Input('first', optional=True),
                        io.Image.Input('last', optional=True), io.Custom('FREEVIDEO_REFERENCES').Input('references', optional=True),
                        io.Custom('FREEVIDEO_MEDIA').Input('media', optional=True), io.Image.Input('smoke_image', optional=True)],
                outputs=[io.Conditioning.Output(), io.Latent.Output()])

        @classmethod
        def execute(cls, clip, vae, audio_vae, variant, text, width, height, frames,
                    first=None, last=None, references=None, media=None, smoke_image=None):
            try:
                from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
            except ImportError as error:
                raise _runtime_error() from error
            first, last, rows = _media(first, last, references, media)
            _check_media(variant, first is not None or last is not None, bool(rows))
            if variant == 'fl2va':
                return MiniMaxH3ImageToVideo.execute(clip=clip, vae=vae, prompt=text, width=width, height=height,
                                                     length=frames, first_frame=first, last_frame=last)
            if smoke_image is not None:
                rows.append({'kind': 'image', 'value': smoke_image})
            return MiniMaxH3ReferenceToVideo.execute(clip=clip, vae=vae, audio_vae=audio_vae, prompt=text,
                width=width, height=height, length=frames, **_references(rows, frames / 24.))

    class ReleaseSampler(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=RELEASE_SAMPLER, display_name='FreeVideo H3 sampling handoff', category='FreeVideo',
                inputs=[io.Latent.Input('latent'), io.Model.Input('model'), io.Clip.Input('clip')],
                outputs=[io.Latent.Output()])

        @classmethod
        def execute(cls, latent, model, clip):
            from comfy import model_management as memory

            # Let ClIP/DiT release their GPU and pinned host allocations before both VAE branches.
            for patcher in (model, clip.patcher):
                memory.unload_model_and_clones(patcher)
                patcher.partially_unload_ram(float('inf'))
            memory.soft_empty_cache()
            memory.throw_exception_if_processing_interrupted()
            return io.NodeOutput(latent)

    return Checkpoint, Conditioning, ReleaseSampler


def _progress_event(server, event, data):
    if event not in ('executing', 'progress', 'execution_error') or not isinstance(data, dict):
        return data
    with _LOCK:
        owners = tuple(_ACTIVE.get(data.get('prompt_id'), ()))
    node = data.get('node_id' if event == 'execution_error' else 'node')
    if not owners or not isinstance(node, str):
        return data
    owner = next((item for item in sorted(owners, key=len, reverse=True) if node.startswith(f'freevideo_h3_{item}_')), None)
    if owner is None:
        return data
    from .comfy_progress import publish
    stage = node[len(f'freevideo_h3_{owner}_'):]
    if event == 'execution_error':
        return {**data, 'node_id': owner, 'native_h3_node_id': node}
    if event == 'executing' and stage == 'condition':
        publish(server, owner, {'label': 'Encoding prompt', 'phase': 'encoding'})
    elif event == 'executing' and stage == 'release_sampling':
        publish(server, owner, {'label': 'Decoding video and audio', 'phase': 'decode'})
    elif event == 'progress' and stage == 'sample':
        value, maximum = data.get('value'), data.get('max')
        if type(value) is int and type(maximum) is int and 0 <= value <= maximum and maximum > 0:
            publish(server, owner, {'label': 'Sampling', 'phase': 'sampling', 'stage': 'single',
                                    'done': value, 'total': maximum, 'unit': 'step'})
    return data


async def register_runtime(lookup=None):
    """Use ComfyUI's prompt lifecycle instead of installing a second executor."""
    global _LOOKUP
    from server import PromptServer
    server = PromptServer.instance
    if lookup is not None:
        _LOOKUP = lookup
    if getattr(server, '_freevideo_h3_lifecycle', None) is not None:
        return runtime_nodes()
    try:
        from comfy_api.latest import Caching
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
        api = Caching()
        parent = api.CacheProvider
        if not all(callable(cls.execute) for cls in (MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo)):
            raise TypeError('Missing native MiniMax H3 nodes')
        classes = runtime_nodes()
    except (AttributeError, ImportError, TypeError) as error:
        raise _runtime_error() from error

    class Lifecycle(parent):
        def should_cache(self, context, value=None):
            return False

        async def on_lookup(self, context):
            return None

        async def on_store(self, context, value):
            return None

        def on_prompt_end(self, prompt_id):
            _release(prompt_id)

    provider = Lifecycle()
    await api.register_provider(provider)
    server._freevideo_h3_lifecycle = provider
    send = server.send_sync

    def send_sync(event, data, sid=None):
        try:
            data = _progress_event(server, event, data)
        except Exception:
            _LOG.debug('Native H3 could not forward ComfyUI progress.', exc_info=True)
        return send(event, data, sid)

    server.send_sync = send_sync
    return classes


def _editable_prompt(prompt):
    if not isinstance(prompt, dict):
        return prompt
    result = {}
    for identity, row in prompt.items():
        metadata = row.get('_meta') if isinstance(row, dict) else None
        owner = metadata.get('freevideo_h3_owner') if isinstance(metadata, dict) else None
        if isinstance(owner, str) and owner in prompt and _owned_node(identity, row, owner):
            continue
        if isinstance(row, dict) and row.get('class_type') == 'FreeVideoGenerate' and isinstance(row.get('inputs'), dict):
            row = dict(row, inputs={key: value for key, value in row['inputs'].items() if key not in (VIDEO, ERROR)})
        result[identity] = row
    return result


def save_native_video(video, output_root, *, preset, prompt, width, height, seconds, seed, steps=20,
                      metadata=None, graph=None, encoder_component=None):
    """Publish the standard FreeVideo work from a native ComfyUI VIDEO result."""
    from comfy_api.latest import Types
    from .file_ops import publish

    profile = _profile(preset, encoder_component)
    folder = Path(output_root).resolve() / 'FreeVideo' / time.strftime('%Y-%m-%d', time.gmtime()) / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / 'video.mp4'
    temporary = folder / 'video.pending.mp4'
    summary = folder / 'video.request.json'
    summary_pending = folder / 'video.request.pending.json'
    metadata = dict(metadata or {})
    if graph is not None and graph.get('prompt') is not None:
        metadata['prompt'] = _editable_prompt(graph['prompt'])
    if graph is not None and graph.get('workflow') is not None:
        metadata['workflow'] = graph['workflow']
    try:
        video.save_to(str(temporary), format=Types.VideoContainer('mp4'), codec=Types.VideoCodec('h264'), metadata=metadata)
        actual_width, actual_height = video.get_dimensions()
        frames = round(video.get_duration() * 24)
        if not temporary.is_file() or temporary.stat().st_size == 0 or frames <= 0:
            raise RuntimeError('ComfyUI did not produce the native H3 video.')
        assets = profile.get('assets') or profile
        checkpoint = assets.get('checkpoint', assets.get('model'))
        from .local_comfy_h3_catalog import DEPENDENCIES
        encoder_name, _, encoder_hash = DEPENDENCIES['encoder']
        report = {'success': True, 'backend': 'comfy_h3', 'native_h3_preset': preset, 'native_h3_variant': _variant(profile),
                  'model_id': preset, 'model_label': profile.get('label') or preset,
                  'model_checkpoint': Path(checkpoint).name if checkpoint else None, 'model_format': profile.get('format'),
                   'text_encoder': profile.get('shared_encoder') or {'id': 'builtin', 'label': encoder_name, 'sha256': encoder_hash},
                  'prompt': prompt, 'seed': seed, 'geometry': {'width': actual_width, 'height': actual_height,
                      'frames': frames, 'seconds': frames / 24., 'fps': 24, 'requested_width': width,
                      'requested_height': height, 'requested_seconds': seconds}, 'output_frames': frames,
                  'sampling_plan': {'total_steps': steps, 'base_steps': steps, 'refine_steps': 0,
                                    'first_steps': steps, 'enabled': False, 'two_pass': False}}
        summary_pending.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
        publish(temporary, output)
        publish(summary_pending, summary)
    except BaseException:
        try:
            for path in (temporary, summary_pending, output):
                path.unlink(missing_ok=True)
            folder.rmdir()
        except OSError:
            _LOG.warning('Native H3 could not remove a failed output.', exc_info=True)
        raise
    return output, report
