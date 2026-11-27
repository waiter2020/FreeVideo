"""Audio model layouts and smoke checks shared with their loaders."""
from functools import partial
import os
from pathlib import Path

from .model_components import Slot, selected_files


MUSIC_DIT = 'acestep-v15-xl-turbo'
MUSIC_BF16 = MUSIC_DIT + '-bf16'
MUSIC_LM = 'acestep-5Hz-lm-1.7B'
MUSIC_TEXT = 'Qwen3-Embedding-0.6B'
MUSIC_SLOTS = {
    'acestep-dit': (MUSIC_BF16 + '/transformer', ('diffusion_pytorch_model-00001-of-00002.safetensors',
                                                 'diffusion_pytorch_model-00002-of-00002.safetensors')),
    'acestep-conditioning': (MUSIC_BF16 + '/condition_encoder', ('diffusion_pytorch_model.safetensors',)),
    'acestep-detokenizer': (MUSIC_DIT, ('config.json', 'model-00004-of-00004.safetensors', 'silence_latent.pt')),
    'acestep-lm': (MUSIC_LM, ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')),
    'acestep-text-encoder': (MUSIC_TEXT, ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')),
    'acestep-audio-vae': ('vae', ('diffusion_pytorch_model.safetensors', 'config.json')),
}
VOICE_SLOTS = {
    'cosyvoice-llm': ('llm.pt', 'CosyVoice-BlankEN/config.json', 'CosyVoice-BlankEN/generation_config.json',
                      'CosyVoice-BlankEN/merges.txt', 'CosyVoice-BlankEN/tokenizer_config.json', 'CosyVoice-BlankEN/vocab.json'),
    'cosyvoice-flow': ('flow.pt',),
    'cosyvoice-hift': ('hift.pt',),
    'cosyvoice-speech-tokenizer': ('speech_tokenizer_v3.onnx',),
    'campplus': ('campplus.onnx',),
}
SUBTITLE_SLOTS = {'en': 'subtitle-nemo-ctc', 'zh': 'subtitle-zipformer-ctc', 'sensevoice': 'subtitle-sensevoice'}
SUBTITLE_LID_FILES = ('tiny-encoder.int8.onnx', 'tiny-decoder.int8.onnx')
SUBTITLE_VAD_FILE = 'silero_vad_v5.onnx'


def overrides(root, identities):
    return {slot: files for slot in identities if (files := selected_files(root, slot)[1])}


def music_paths(base, installed):
    paths = {}
    for slot, (directory, names) in MUSIC_SLOTS.items():
        files = installed.get(slot, {})
        paths.update({directory + '/' + name: files.get(name, base / directory / name) for name in names})
    return paths


def _validate_music(root, defaults, slot, folder):
    import torch
    from .music import pipeline
    installed = overrides(root, (component for component in MUSIC_SLOTS if component != slot))
    installed[slot] = {name: folder / name for name in MUSIC_SLOTS[slot][1]}
    paths = music_paths(defaults, installed)
    kind = {'acestep-lm': 'lm', 'acestep-text-encoder': 'text', 'acestep-audio-vae': 'vae'}.get(slot, 'dit')
    if kind == 'lm':
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        model_path = paths[MUSIC_LM + '/model.safetensors']
        with torch.device('meta'):
            model = Qwen3ForCausalLM(Qwen3Config.from_pretrained(model_path.parent, local_files_only=True))
        weights = pipeline._weight_map(model_path)
        if model.config.tie_word_embeddings and 'lm_head.weight' not in weights:
            weights['lm_head.weight'] = weights['model.embed_tokens.weight']
        pipeline._load_weights(model, weights)
        tokenizer = AutoTokenizer.from_pretrained(model_path.parent, local_files_only=True, trust_remote_code=False)
        template = tokenizer.apply_chat_template([dict(role='user', content='music')], tokenize=False, add_generation_prompt=True)
        from .music.constrained_logits import MetadataConstrainedLogitsProcessor
        if 'music' not in template:
            raise ValueError('The tokenizer needs the ACE-Step chat template.')
        processor = MetadataConstrainedLogitsProcessor(tokenizer, max_duration=600)
        if not processor.audio_code_token_ids:
            raise ValueError('The tokenizer must provide ACE-Step audio code tokens.')
    else:
        files = {'text': MUSIC_TEXT + '/model.safetensors', 'vae': 'vae/diffusion_pytorch_model.safetensors'}
        weights = pipeline._dit_weight_map(defaults, paths) if kind == 'dit' else pipeline._weight_map(paths[files[kind]])
        model = pipeline._construct(kind, defaults, weights, 'meta', torch.float32, paths=paths)
        from .music.acestep_model import AceStepConfig
        contract = AceStepConfig()
        if kind == 'dit' and (model.config.text_hidden_dim != contract.text_hidden_dim
                              or model.config.in_channels != contract.in_channels):
            raise ValueError('The DiT does not match the ACE-Step conditioning and latent widths.')
        if kind == 'text' and model.config.hidden_size != contract.text_hidden_dim:
            raise ValueError('The text encoder does not match the ACE-Step conditioning width.')
        if kind == 'vae' and model.config.get('latent_channels') != 64:
            raise ValueError('The audio VAE must accept 64-channel ACE-Step latents.')
        if kind == 'text':
            from transformers import PreTrainedTokenizerFast
            PreTrainedTokenizerFast.from_pretrained(folder, local_files_only=True)('music')
        if slot == 'acestep-detokenizer':
            silence = torch.load(folder / 'silence_latent.pt', map_location='cpu', weights_only=True)
            if not isinstance(silence, torch.Tensor) or silence.ndim != 3 or silence.shape[:2] != (1, 64) or silence.shape[2] < 750:
                raise ValueError('The ACE-Step silence latent has an incompatible shape.')


def _voice_networks(folder, kind, original):
    import torch
    from .dubbing.cosyvoice3.build import QWEN_DIR, build_networks, get_qwen_tokenizer
    qwen = folder / QWEN_DIR if kind == 'llm' else original / QWEN_DIR
    with torch.device('meta'):
        model = build_networks(str(qwen))[('llm', 'flow', 'hift').index(kind)]
    tensors = torch.load(folder / (kind + '.pt'), map_location='meta', weights_only=True)
    if kind == 'hift':
        tensors = {key.replace('generator.', ''): value for key, value in tensors.items()}
    model.load_state_dict(tensors, strict=True)
    if kind == 'llm':
        get_qwen_tokenizer(token_path=str(qwen), skip_special_tokens=True, version='cosyvoice3').encode('test')


def _onnx_voice(folder, file, model_root):
    import numpy as np
    from .dubbing import pack
    pack.activate(pack.runtime_path(model_root))
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    model = onnxruntime.InferenceSession(str(folder / file), sess_options=options, providers=['CPUExecutionProvider'])
    features, *extra = model.get_inputs()
    if file == 'campplus.onnx':
        result = model.run(None, {features.name: np.zeros((1, 100, 80), dtype='float32')})[0]
        if result.shape != (1, 192) or not np.isfinite(result).all():
            raise ValueError('CAM++ must return a finite 192-dimensional speaker embedding.')
    else:
        length, = extra
        result = model.run(None, {features.name: np.zeros((1, 128, 100), dtype='float32'),
                                  length.name: np.asarray([100], dtype='int32')})[0]
        if not result.size or result.min() < 0 or result.max() >= 6561:
            raise ValueError('The speech tokenizer returned incompatible token IDs.')


def _subtitle(folder, kind):
    import numpy as np
    from . import subtitle_pack, subtitle_sherpa
    subtitle_pack.activate()
    if kind == 'vad':
        detector = subtitle_sherpa.make_vad(folder / subtitle_sherpa.VAD_MODEL)
        detector.accept_waveform(np.zeros(1024, dtype='float32'))
    elif kind == 'lid':
        recognizer = subtitle_sherpa.make_lid(folder, 1)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, np.zeros(16000, dtype='float32'))
        recognizer.compute(stream)
    else:
        language = 'zh' if kind == 'zh' else 'en'
        engine = subtitle_sherpa.Engine({language: folder}, 1,
                                        multilingual_folder=folder if kind == 'sensevoice' else None)
        recognizer = engine._recognizer(language)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, np.zeros(16000, dtype='float32'))
        recognizer.decode_stream(stream)


def _music_vad(folder):
    import numpy as np
    from .speech import probabilities
    result = probabilities(np.zeros(16000, dtype='float32'), folder / 'model.onnx')
    if not result.size or not np.isfinite(result).all():
        raise ValueError('The speech detector failed its input and output check.')


def _stems(folder):
    from .stems.separate import load_model
    load_model(folder / 'bandit_plus_cocktail.pt', device='cpu')


def _installer(install):
    def run(progress, check):
        def moved(value):
            if value.get('total'):
                progress(value.get('done', 0), value['total'])
        return install(moved, check)
    return run


def _install_voice(root, files, repair, progress, check):
    from .dubbing import pack
    from .locking import runtime_lock
    with runtime_lock(root / 'dubbing-install.lock', inherit=False):
        pack.install(root, progress, check, only=files, repair=repair)


def slots(root, machine):
    from . import subtitle_pack
    from .dubbing import pack
    from .image_gen import image_pack

    model_root = Path(machine.get('model_root') or root / 'models')
    image_root = image_pack.base(model_root)
    music_root = image_root / 'music'
    dubbing_root = root
    voice_root = pack.directory(dubbing_root)
    subtitle_models = Path(os.environ.get('FREEVIDEO_MODEL_ROOT') or root / 'models')
    subtitle_root = subtitle_pack.base(subtitle_models)
    names = {
        'acestep-dit': ('ACE-Step 1.5 XL Turbo DiT', 'ACE-Step 1.5 XL Turbo 去噪器'),
        'acestep-conditioning': ('ACE-Step conditioning encoder', 'ACE-Step 条件编码器'),
        'acestep-detokenizer': ('ACE-Step audio detokenizer', 'ACE-Step 音频标记解码器'),
        'acestep-lm': ('ACE-Step 5 Hz 1.7B language model', 'ACE-Step 5 Hz 1.7B 语言模型'),
        'acestep-text-encoder': ('Qwen3-Embedding-0.6B', 'Qwen3-Embedding-0.6B 文本编码器'),
        'acestep-audio-vae': ('ACE-Step Oobleck audio VAE', 'ACE-Step Oobleck 音频 VAE'),
        'cosyvoice-llm': ('Fun-CosyVoice3-0.5B-2512 language model', 'Fun-CosyVoice3-0.5B-2512 语言模型'),
        'cosyvoice-flow': ('CosyVoice3 flow', 'CosyVoice3 流匹配模型'),
        'cosyvoice-hift': ('CosyVoice3 HiFT vocoder', 'CosyVoice3 HiFT 声码器'),
        'cosyvoice-speech-tokenizer': ('CosyVoice3 speech tokenizer v3', 'CosyVoice3 语音标记器 v3'),
        'campplus': ('CAM++ speaker encoder', 'CAM++ 说话人编码器'),
    }
    cuda = machine.get('device_backend', 'cuda') != 'mps'
    image_sizes = {row['file']: row['bytes'] for group in ('music', 'stems') for row in image_pack.rows(group=group)}

    def image_slot(identity, label, zh, files, validate, group, consumers, supported=True):
        only = {path.relative_to(image_root).as_posix() for path in files.values()}
        return Slot(id=identity, group='audio', label=label, label_zh=zh, consumers=consumers, files=files,
                    format='ONNX' if all(name.endswith('.onnx') for name in files) else 'PyTorch / Safetensors',
                    validate=validate, supported=supported, reason='' if supported else 'Requires CUDA or ROCm.',
                    install_default=_installer(lambda progress, check: image_pack.install(
                        model_root, progress, check, engine_root=root, group=group, only=only)),
                    repair_default=_installer(lambda progress, check: image_pack.install(
                        model_root, progress, check, engine_root=root, group=group, only=only, repair=True)),
                    default_bytes=sum(image_sizes.get(name, 0) for name in only),
                    source='ACE-Step / FreeVideo' if group == 'music' else 'Bandit Plus / FreeVideo')

    result = [image_slot(slot, *names[slot], {name: music_root / directory / name for name in files},
                         partial(_validate_music, root, music_root, slot), 'music', ('Music generation',), cuda)
              for slot, (directory, files) in MUSIC_SLOTS.items()]
    result += [image_slot('editor-silero-vad', 'Silero VAD', 'Silero VAD 编辑器语音检测',
                          {'model.onnx': music_root / 'silero-vad/model.onnx'}, _music_vad, 'music', ('Lower music during speech',)),
               image_slot('bandit-plus', 'Bandit Plus', 'Bandit Plus 音轨分离',
                          {'bandit_plus_cocktail.pt': image_root / 'stems/bandit_plus_cocktail.pt'}, _stems,
                          'stems', ('Separate dialogue, music and ambience',), cuda)]
    voice_sizes = {row['file']: row['bytes'] for row in pack.rows()}
    voice_runtime = sum(row['bytes'] for row in pack.wheel_rows() or ())
    for slot, files in VOICE_SLOTS.items():
        kind = slot.removeprefix('cosyvoice-')
        validate = (partial(_voice_networks, kind=kind, original=voice_root) if kind in ('llm', 'flow', 'hift')
                    else partial(_onnx_voice, file=files[0], model_root=dubbing_root))
        result.append(Slot(id=slot, group='audio', label=names[slot][0], label_zh=names[slot][1],
                           consumers=('Dubbing', 'Speaker identification') if slot == 'campplus' else ('Dubbing',),
                           files={name: voice_root / name for name in files}, format='ONNX' if files[0].endswith('.onnx') else 'PyTorch',
                           validate=validate, source='FunAudioLLM/Fun-CosyVoice3-0.5B-2512', license='Apache-2.0',
                           install_default=_installer(partial(_install_voice, dubbing_root, files, False)),
                           repair_default=_installer(partial(_install_voice, dubbing_root, files, True)),
                           default_status=lambda files=files: dict(download_bytes=sum(voice_sizes.get(name, 0) for name in files)
                                                                  + (0 if pack.runtime_path(dubbing_root) else voice_runtime)),
                           default_bytes=sum(voice_sizes.get(name, 0) for name in files)))
    subtitle_specs = (
        (SUBTITLE_SLOTS['en'], 'NeMo Conformer CTC Small', 'NeMo Conformer CTC Small 英语字幕', 'en', 'en', 'speech', ('model.int8.onnx', 'tokens.txt')),
        (SUBTITLE_SLOTS['zh'], 'Chinese Zipformer CTC', 'Zipformer CTC 中文字幕', 'zh', 'zh', None, ('model.int8.onnx', 'tokens.txt')),
        (SUBTITLE_SLOTS['sensevoice'], 'SenseVoiceSmall', 'SenseVoiceSmall 多语字幕', 'sensevoice', 'sensevoice', 'language', ('model.int8.onnx', 'tokens.txt')),
        ('subtitle-whisper-lid', 'Whisper Tiny language identification', 'Whisper Tiny 字幕语种识别', 'lid', 'lid', 'language', SUBTITLE_LID_FILES),
        ('subtitle-silero-vad', 'Silero VAD v5', 'Silero VAD v5 字幕语音检测', '', 'vad', 'speech', (SUBTITLE_VAD_FILE,)),
    )
    for slot, label, zh, directory, kind, group, names_ in subtitle_specs:
        files = {name: subtitle_root / directory / name for name in names_}
        expected = {row['file']: row['bytes'] for row in subtitle_pack.rows(group)} if group else {}
        result.append(Slot(id=slot, group='audio', label=label, label_zh=zh, consumers=('Subtitles',), files=files,
                           format='ONNX', validate=partial(_subtitle, kind=kind), source='FreeVideo subtitle pack' if group else 'Local model library',
                           install_default=_installer(partial(subtitle_pack.install, ('runtime', group), subtitle_models)) if group else None,
                           default_status=(lambda group=group: dict(download_scope='Subtitles', download_bytes=sum(
                               row['bytes'] for family in ('runtime', group) for row in subtitle_pack.rows(family)))) if group else None,
                           default_bytes=sum(expected.get(path.relative_to(subtitle_root).as_posix(), 0) for path in files.values())))
    return result
