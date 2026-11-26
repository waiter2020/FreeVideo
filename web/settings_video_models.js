import { bar, button, card, checkify, el, field, groupHead, row, setText, slot } from './settings_kit.js';
import { bytesText, setupError, t } from './settings_words.js';
import { userMessage } from './user_message.js';

const TRANSFERRING = new Set(['downloading', 'copying']);
const RUNNING = new Set([...TRANSFERRING, 'preparing', 'verifying', 'waiting_for_gpu', 'smoke']);
const PAUSED = new Set(['failed', 'cancelled']);
const PHASES = {
    downloading: ['Downloading model files…', '正在下载模型文件…'],
    copying: ['Copying model files…', '正在复制模型文件…'],
    preparing: ['Preparing model files…', '正在准备模型文件…'],
    verifying: ['Verifying model files…', '正在校验模型文件…'],
    waiting_for_gpu: ['Waiting for the GPU to become idle…', '正在等待 GPU 空闲…'],
    smoke: ['Testing the model on this GPU…', '正在本机 GPU 上试运行模型…'],
    ready: ['Installed. Select Use model to activate it.', '已安装。点击“使用模型”即可启用。'],
    failed: ['Installation stopped. Resume to try again.', '安装中断。点击“继续下载”可重试。'],
    cancelled: ['Paused. Downloaded files are kept.', '已暂停，已下载的文件已保留。'],
};
const LOCAL_END = {
    failed: ['Import stopped. Select Retry copy to try again.', '导入中断。点击“重试复制”可重试。'],
    cancelled: ['Import cancelled. Select Retry copy to start again.', '导入已取消。点击“重试复制”可重新开始。'],
};
const ACTIVATION_END = {
    failed: ['Verification failed. The previous component is still in use.', '验证失败，当前仍使用原组件。'],
    cancelled: ['Verification cancelled. The previous component is still in use.', '验证已取消，当前仍使用原组件。'],
};
const VALIDATION = {
    text: ['Text to image', '纯文字出图'],
    reference: ['Reference image', '参考图出图'],
    mask: ['Masked edit', '遮罩编辑'],
    fl2va: ['FL2VA', 'FL2VA'],
    ref2va: ['Ref2VA', 'Ref2VA'],
};
const LOCAL_SOURCE = ['Local import', '本地导入'];
const localModel = model => model?.source_kind === 'local';
const formatName = format => ({comfy_int8: 'ComfyUI INT8', comfy_float: t('ComfyUI floating point', 'ComfyUI 浮点'),
    int8_convrot: 'INT8 ConvRot', nvfp4: 'NVFP4', nvfp4_awq: 'NVFP4 AWQ', bf16: 'BF16', fp16: 'FP16', fp32: 'FP32'})[format] || format;
const comfyPresetName = preset => preset === 'ref2va' ? 'Ref2VA' : preset?.toUpperCase();
const VIDEO = {
    route: 'video-model', response: 'video_models', icon: 'film',
    heading: ['Video models', '视频模型'], progress: ['Video model import progress', '视频模型导入进度'],
    switching: ['Switching video model…', '正在切换视频模型…'],
    incompatible: ['This model is incompatible with the installed video engine.', '此模型与已安装的视频引擎不兼容。'],
};
const IMAGE = {
    route: 'image-model', response: 'image_models', icon: 'img',
    heading: ['Picture model switching', '图像模型切换'], progress: ['Picture model import progress', '图像模型导入进度'],
    switching: ['Switching picture model…', '正在切换图像模型…'],
    incompatible: ['This model is incompatible with the selected picture model family.', '此模型与所选图像模型系列不兼容。'],
};
const IMAGE_COMPONENTS = {
    dit: ['Denoiser', '去噪器'],
    text_encoder: ['Text encoder', '文本编码器'],
    vae: ['VAE', 'VAE'],
};
const QWEN_ENCODER = {
    model: 'qwen-image-2.1-turbo',
    filename: 'qwen3vl_8b_int8_convrot.safetensors',
    official: 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/6e8c3331a4dcb0c125e94ad6a7cb6b850149be39/text_encoders/qwen3vl_8b_int8_convrot.safetensors',
    examples: [
        ['pottokao', 'qwen3vl_8b_int8_convrot_heretic.safetensors', 'https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot/blob/033b4bf23f8816b22c065309bc9b349953193978'],
        ['NidAll', 'qwen3vl_8b_abliterated_int8_convrot.safetensors', 'https://huggingface.co/NidAll/Qwen3VL-8B-Abliterated-INT8-ConvRot/blob/1e554d2e4b2efec4d256db80b7b6879b8a8d79a3'],
        ['paralaif', 'Huihui-Qwen3-VL-8B-Instruct-abliterated_int8_convrot.safetensors', 'https://huggingface.co/paralaif/Huihui-Qwen3-VL-8B-Instruct-abliterated-comfy-int8-convrot/blob/1c0cae0033990848b3c61b113f83b81b9ad4d7b2'],
    ],
};
const isQwenEncoder = (model, component) => model === QWEN_ENCODER.model && component === 'text_encoder';
const IMAGE_FORMATS = {
    'qwen-image-2.1-turbo': {
        dit: ['Qwen-Image 2.1 Turbo INT8 ConvRot, one .safetensors file.', 'Qwen-Image 2.1 Turbo INT8 ConvRot，单个 .safetensors 文件。'],
        text_encoder: ['Qwen3-VL 8B, INT8 ConvRot text weights (group size 256) and a complete BF16 vision tower in one .safetensors file.',
            'Qwen3-VL 8B，INT8 ConvRot 文本权重（组大小 256）和完整 BF16 视觉塔须在同一个 .safetensors 文件中。'],
        vae: ['Qwen-Image 2.1 BF16 VAE, one .safetensors file.', 'Qwen-Image 2.1 BF16 VAE，单个 .safetensors 文件。'],
    },
    'minimax-h3-image': {
        dit: ['MiniMax H3 Image INT8 ConvRot, one .safetensors file.', 'MiniMax H3 Image INT8 ConvRot，单个 .safetensors 文件。'],
        text_encoder: ['MiniMax H3 Qwen3-VL 32B in ComfyUI NVFP4 AWQ or plain NVFP4 format, with all 50 language layers and the full vision tower in one .safetensors file. H3 Image, VDN, and managed FL2VA/Ref2VA share this encoder.',
            '支持 ComfyUI 格式的 MiniMax H3 Qwen3-VL 32B NVFP4 AWQ 或普通 NVFP4。单个 .safetensors 文件须包含全部 50 个语言层和完整视觉塔。H3 Image、VDN 和托管 FL2VA／Ref2VA 共用此编码器。'],
        vae: ['MiniMax H3 Image FP16 VAE, one .safetensors file.', 'MiniMax H3 Image FP16 VAE，单个 .safetensors 文件。'],
    },
    'moss-image-sft': {
        dit: ['MOSS-Image-SFT BF16 denoiser, a .safetensors file or a Diffusers shard index.',
            'MOSS-Image-SFT BF16 去噪器，支持 .safetensors 文件或 Diffusers 分片索引。'],
        text_encoder: ['MOSS-Image-SFT BF16 text encoder, a .safetensors file or a Transformers shard index.',
            'MOSS-Image-SFT BF16 文本编码器，支持 .safetensors 文件或 Transformers 分片索引。'],
        vae: ['MOSS-Image-SFT FP32 VAE, one .safetensors file.', 'MOSS-Image-SFT FP32 VAE，单个 .safetensors 文件。'],
    },
};
const componentName = component => t(...(IMAGE_COMPONENTS[component] || IMAGE_COMPONENTS.dit));

function licenceLink(label, href) {
    try {
        const url = new URL(href);
        if (url.protocol === 'https:') {
            const link = el('a', '', label);
            link.href = url.href;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            return link;
        }
    } catch { /* Display the licence as text when there is no link. */ }
    return document.createTextNode(label);
}

function hubTree(repo, revision, path = '') {
    const part = value => String(value || '').split('/').map(encodeURIComponent).join('/');
    return `https://huggingface.co/${part(repo)}/tree/${encodeURIComponent(revision)}${path ? '/' + part(path) : ''}`;
}

export function hfModelSettings({request, refresh, kind = 'video', onActivate}) {
    const isImage = kind === 'image';
    const words = isImage ? IMAGE : VIDEO;
    const heading = groupHead(t(...words.heading));
    const installed = card();
    const source = el('form', 'st-sub');
    const family = isImage ? el('select') : null;
    const component = isImage ? el('select') : null;
    const formatNote = isImage ? el('p', 'st-p') : null;
    const encoderFiles = isImage ? el('div') : null;
    const familyNote = isImage ? el('p', 'st-p', t('Install a base picture model above before adding weights for that family.',
        '请先在上方安装基础图像模型，再为该系列添加权重。')) : null;
    if (family) {
        const familyLabel = el('label', 'st-label');
        familyLabel.append(el('span', '', t('Base model', '基础模型')), field(family));
        const componentLabel = el('label', 'st-label');
        componentLabel.append(el('span', '', t('Component', '组件')), field(component));
        const official = el('p', 'st-p', t('Built-in checkpoint ', '内置权重 '));
        official.append(licenceLink(QWEN_ENCODER.filename, QWEN_ENCODER.official));
        const community = el('ul', 'st-p');
        for (const [owner, file, url] of QWEN_ENCODER.examples) {
            const item = el('li', 'st-path');
            item.append(licenceLink(`${owner} · ${file}`, `${url}/${file}`));
            community.append(item);
        }
        encoderFiles.append(official, el('p', 'st-p', t('Community checkpoints that passed the file check', '通过文件初检的社区权重')),
            community, el('p', 'st-p', t('Activation tests text, reference pictures and masked edits on your GPU.',
            '启用时会在本机试跑纯文出图、参考图和遮罩编辑。')));
        encoderFiles.hidden = true;
        source.append(familyLabel, componentLabel, formatNote, encoderFiles, familyNote);
    }
    const label = el('label', 'st-label');
    label.append(el('span', '', t('Hugging Face model', 'Hugging Face 模型')));
    const input = el('input');
    input.type = 'text';
    input.spellcheck = false;
    input.autocomplete = 'off';
    input.maxLength = 512;
    input.placeholder = t('Search name, repository ID or Hugging Face link', '搜索名称、仓库 ID 或 Hugging Face 链接');
    label.append(field(input));
    const search = button(t('Search', '搜索'), {iconName: 'search', kind: 'fill'});
    const inspect = button(t('Check compatibility', '检查兼容性'), {iconName: 'check', kind: 'fill'});
    const tools = el('div', 'st-actions st-video-model-actions');
    tools.append(search, inspect);
    source.append(label, tools);

    const localSource = el('div', 'st-sub');
    localSource.hidden = true;
    const localTitle = el('div', 'st-rt');
    localTitle.append(el('b', '', t('Import from this computer', '从本地导入')));
    const localActions = el('div', 'st-actions st-video-model-actions');
    const localFile = button(isImage ? t('Choose local file', '选择本地文件') : t('Choose cache manifest', '选择缓存清单'),
        {iconName: 'folder', kind: 'fill'});
    const localFolder = isImage ? null : button(t('Choose source weights folder', '选择原始权重目录'), {iconName: 'folder', kind: 'fill'});
    const comfyFolder = isImage ? null : button(t('Scan ComfyUI H3 folder', '扫描 ComfyUI H3 权重目录'), {iconName: 'search', kind: 'fill'});
    localActions.append(...[localFile, localFolder, comfyFolder].filter(Boolean));
    localSource.append(localTitle, localActions);
    const comfyResults = isImage ? null : el('div', 'st-video-model-results st-video-model-comfy');
    if (comfyResults) {
        comfyResults.hidden = true;
        comfyResults.setAttribute('aria-label', t('ComfyUI H3 weights', 'ComfyUI H3 权重'));
        localSource.append(comfyResults);
    }

    const results = el('div', 'st-sub st-video-model-results');
    results.setAttribute('aria-label', t('Search results', '搜索结果'));
    const review = el('div', 'st-sub');
    const candidateName = el('div', 'st-rt');
    const candidateTitle = el('b');
    candidateName.append(candidateTitle);
    const facts = el('dl', 'st-facts st-video-model-facts');
    const compatibility = slot();
    const variants = el('div', 'st-video-model-variants');
    const agreement = el('label', 'st-agree');
    const accept = checkify(el('input'));
    accept.type = 'checkbox';
    agreement.append(accept, el('span', '', t('I accept this download and the model licence.', '我同意下载并接受模型许可证。')));
    const install = button(t('Download model', '下载模型'), {iconName: 'down', kind: 'primary'});
    const copyModel = button(t('Copy and verify', '复制并验证'), {iconName: 'copy', kind: 'primary'});
    const decline = button(t('Close review', '关闭预览'));
    const reviewActions = el('div', 'st-actions st-video-model-actions');
    reviewActions.append(install, copyModel, decline);
    review.append(candidateName, facts, compatibility, variants, agreement, reviewActions);

    const progress = el('div', 'st-sub');
    const status = slot();
    const meter = bar();
    meter.setAttribute('aria-label', t(...words.progress));
    const pause = button(t('Pause', '暂停'), {iconName: 'stop', kind: 'fill'});
    const resume = button(t('Resume download', '继续下载'), {iconName: 'down', kind: 'fill'});
    const progressActions = el('div', 'st-actions st-video-model-actions');
    progressActions.append(pause, resume);
    progress.append(status, meter, progressActions);
    const notice = slot('st-under');
    const panel = card(source, localSource, results, review, progress);
    const elements = [heading, installed, panel, notice];
    elements.forEach(element => { element.hidden = true; });
    results.hidden = true;
    review.hidden = true;
    progress.hidden = true;

    let view = {};
    let candidate = null;
    let requesting = false;
    let setupBusy = false;
    let installedKey;
    let familyKey;
    let componentKey;
    let announcedActivation;
    let reviewTrigger = input;
    let comfyRows = [];
    const task = () => view.task || {};
    const phase = () => task().phase || task().status || 'idle';
    const busy = () => RUNNING.has(phase());
    const blocked = () => setupBusy || busy() || requesting;
    const localAvailable = () => view.local_available === true;
    const comfyAvailable = () => localAvailable() && view.comfy_h3_available !== false;
    const localTask = () => localModel(task()) || localModel(task().candidate);
    const activating = () => task().operation === 'activate';
    function installedModels() {
        const models = new Map((view.installed || []).map(model => [model.id, model]));
        for (const family of Object.values(view.components || {})) {
            for (const role of Object.values(family)) {
                for (const model of role.installed || []) models.set(model.id, model);
                if (role.active?.damaged) models.set(role.active.id, role.active);
            }
        }
        return [...models.values()];
    }
    const hasCandidate = () => installedModels().some(model => model.id === candidate?.id);
    const families = () => view.families || [];
    const roles = model => ({dit: {ready: model.ready}, ...view.components?.[model.id]});
    const modelReady = model => Object.values(roles(model)).some(role => role?.ready);
    const selection = () => family ? {model: family.value, component: component.value || 'dit'} : {};
    const modelSelection = model => isImage ? {model: model.model, component: model.component || 'dit'} : {};
    const componentReady = () => !family || families().some(model => model.id === family.value && roles(model)[component.value]?.ready);
    function isActive(model) {
        if (!isImage) return view.active?.id === model.id;
        const role = model.component || 'dit';
        const active = view.components?.[model.model]?.[role]?.active || (role === 'dit' ? view.active?.[model.model] : null);
        return model.active === true || active?.id === model.id;
    }
    const defaultName = label => t(`${label} (default)`, `${label}（默认）`);
    const builtinQwenEncoder = model => model.id?.startsWith('official:') && isQwenEncoder(model.model, model.component);
    const nameOf = model => model.id === 'official' ? defaultName('MiniMax H3') :
        builtinQwenEncoder(model) ? defaultName(QWEN_ENCODER.filename) :
        model.id?.startsWith('official:') ? defaultName(model.label) : model.label || model.repo;
    const message = (text, warning = false) => {
        notice.className = `st-slot st-under${warning ? ' is-warn' : ''}`;
        notice.textContent = text;
        notice.hidden = !view.available || !text;
    };

    function controls() {
        const locked = blocked();
        const importing = locked || !componentReady();
        localSource.hidden = !localAvailable();
        localFile.disabled = importing;
        if (localFolder) localFolder.disabled = importing;
        if (comfyFolder) {
            comfyFolder.hidden = !comfyAvailable();
            comfyFolder.disabled = importing;
            comfyResults.hidden = !comfyAvailable() || !comfyResults.childElementCount;
            for (const entry of comfyRows) paintComfyRow(entry, importing);
        }
        search.disabled = inspect.disabled = importing || !input.value.trim();
        input.disabled = importing;
        if (family) {
            family.disabled = locked || !families().some(modelReady);
            component.disabled = locked || !componentReady();
            const hint = IMAGE_FORMATS[family.value]?.[component.value];
            formatNote.textContent = hint ? t(...hint) : '';
            formatNote.hidden = !hint;
            encoderFiles.hidden = !isQwenEncoder(family.value, component.value);
        }
        for (const action of results.querySelectorAll('button')) action.disabled = importing;
        for (const action of installed.querySelectorAll('button')) action.disabled = locked;
        for (const action of variants.querySelectorAll('button')) action.disabled = importing;
        accept.disabled = importing;
        const cannotImport = importing || candidate?.compatible !== true || hasCandidate();
        install.disabled = cannotImport || !accept.checked;
        copyModel.disabled = cannotImport || !localAvailable() || candidate?.kind === 'comfy_h3' && !comfyAvailable();
        pause.disabled = requesting;
        resume.disabled = locked || !activating() && localTask() && !localAvailable();
    }

    async function run(action, payload, started, receive) {
        if (requesting) return;
        requesting = true;
        message(started);
        controls();
        try {
            const result = await request(`${words.route}-${action}`, payload);
            requesting = false;
            message('');
            receive(result);
        } catch (error) {
            message(setupError(error?.message) ?? userMessage(error, t), true);
        } finally {
            requesting = false;
            controls();
        }
    }

    function paintInstalled() {
        const models = installedModels();
        installed.hidden = !view.available || !models.length;
        const key = JSON.stringify([view.active, view.components, models]);
        if (key === installedKey) return;
        installedKey = key;
        installed.replaceChildren(...models.map(model => {
            const active = isActive(model);
            const mark = model.damaged ? el('span', 'st-chip', t('Unavailable', '不可用')) :
                active ? el('span', 'st-chip is-ok', t('In use', '使用中')) :
                button(t('Use model', '使用模型'), {kind: 'fill'});
            if (!active && !model.damaged) mark.onclick = () => activate(model);
            const needsFamily = isImage && (!model.id?.startsWith('official:') || model.component && model.component !== 'dit');
            const base = needsFamily ? families().find(item => item.id === model.model)?.label : '';
            const origin = localModel(model) ? t(...LOCAL_SOURCE) : model.repo;
            const format = builtinQwenEncoder(model) ? t('Qwen3-VL 8B INT8 ConvRot, BF16 vision tower', 'Qwen3-VL 8B INT8 ConvRot，BF16 视觉塔') : formatName(model.format);
            const summary = [base, isImage && componentName(model.component), origin, format, model.kind === 'comfy_h3' && comfyPresetName(model.preset),
                model.revision?.slice(0, 12)].filter(Boolean).join(' · ');
            return row({iconName: words.icon, title: nameOf(model), sub: summary, end: [mark]}).element;
        }));
    }

    function paintFamilies() {
        if (!family) return;
        const models = families().map(model => ({id: model.id, label: model.label, ready: modelReady(model)}));
        const key = JSON.stringify(models);
        const before = family.value;
        if (key !== familyKey) {
            familyKey = key;
            family.replaceChildren(...models.map(model => {
                const option = el('option', '', model.ready ? model.label : t(`${model.label} (install first)`, `${model.label}（需先安装）`));
                option.value = model.id;
                option.disabled = !model.ready;
                return option;
            }));
            family.value = models.some(model => model.id === before && model.ready) ? before : models.find(model => model.ready)?.id || models[0]?.id || '';
            familyNote.hidden = models.some(model => model.ready);
        }
        paintComponents();
        if (family.value !== before) resetSearch();
    }

    function paintComponents() {
        const model = families().find(item => item.id === family.value);
        const options = model ? Object.entries(roles(model)).filter(([id]) => id in IMAGE_COMPONENTS) : [];
        const key = JSON.stringify([family.value, options.map(([id, role]) => [id, !!role?.ready])]);
        if (key === componentKey) return;
        componentKey = key;
        const before = component.value;
        component.replaceChildren(...options.map(([id, role]) => {
            const name = componentName(id);
            const option = el('option', '', role?.ready ? name : t(`${name} (install first)`, `${name}（需先安装）`));
            option.value = id;
            option.disabled = !role?.ready;
            return option;
        }));
        component.value = options.some(([id, role]) => id === before && role?.ready) ? before :
            options.find(([, role]) => role?.ready)?.[0] || options[0]?.[0] || '';
        if (component.value !== before) resetSearch();
    }

    function focusComponent(model, role) {
        if (!family || !families().some(item => item.id === model && roles(item)[role]?.ready)) return;
        heading.scrollIntoView?.({block: 'start'});
        if (requesting) return;
        family.value = model;
        paintComponents();
        component.value = role;
        resetSearch();
        component.focus();
    }

    function showCandidate(model) {
        candidate = model;
        accept.checked = false;
        review.hidden = !candidate;
        if (!candidate) return;
        candidateTitle.textContent = nameOf(model);
        const fromLocal = localModel(model);
        const transfer = (fromLocal ? model.copy_bytes : model.download_bytes) ?? model.bytes;
        const required = model.required_bytes ?? model.bytes;
        const modelPath = model.path || model.subfolder || '';
        const origin = fromLocal ? [[t('Source', '来源'), t(...LOCAL_SOURCE)]] : [
            [t('Repository', '仓库'), model.repo], [t('Commit', '提交'), model.revision], [t('Folder', '目录'), modelPath],
        ];
        const licence = fromLocal ? [] : [[t('Licence', '许可证'), licenceLink(model.license || t('View model card', '查看模型卡片'),
            model.license_url || hubTree(model.repo, model.revision))]];
        const details = [
            ...origin,
            [t('Component', '组件'), isImage && componentName(model.component)],
            [t('Format', '格式'), formatName(model.format)],
            [t('Preset', '预设'), model.kind === 'comfy_h3' && comfyPresetName(model.preset)],
            [t('Files', '文件数'), model.file_count > 0 && String(model.file_count)],
            [fromLocal ? t('Copy size', '复制大小') : t('Download', '下载量'), Number.isFinite(transfer) && bytesText(transfer)],
            [t('Disk space required', '所需磁盘空间'), Number.isFinite(required) && bytesText(required)],
            ...licence,
        ];
        facts.replaceChildren(...details.filter(([, value]) => value).map(([key, value]) => {
            const fact = el('div');
            const detail = el('dd'); detail.append(value);
            fact.append(el('dt', '', key), detail);
            return fact;
        }));
        const alreadyInstalled = hasCandidate();
        const compatible = model.compatible === true;
        compatibility.className = `st-slot ${compatible ? 'is-ok' : 'is-warn'}`;
        compatibility.textContent = alreadyInstalled ? t('Already installed.', '已安装。') : compatible
            ? t('Compatible with this installation.', '与当前安装兼容。')
            : model.compatibility_reason || model.reason || t(...words.incompatible);
        const alternatives = (fromLocal ? [] : model.variants || []).filter(variant => (variant.path || variant.subfolder || '') !== modelPath
            || variant.revision !== model.revision);
        variants.replaceChildren(...alternatives.map(variant => {
            const path = variant.path || variant.subfolder || '';
            const view = button(t('Check', '检查'), {kind: 'fill'});
            view.onclick = () => check(hubTree(model.repo, variant.revision, path));
            return row({title: variant.format, sub: path || t('Repository root', '仓库根目录'), end: [view]}).element;
        }));
        variants.hidden = !alternatives.length;
        agreement.hidden = install.hidden = fromLocal || !compatible || alreadyInstalled;
        copyModel.hidden = !fromLocal || !compatible || alreadyInstalled;
        controls();
    }

    function update(value, installerBusy = setupBusy) {
        setupBusy = installerBusy;
        if (value && !requesting) {
            view = value;
            heading.hidden = panel.hidden = !view.available;
            notice.hidden = !view.available || !notice.textContent;
            paintFamilies();
            paintInstalled();
            const current = task();
            const state = phase();
            const active = installedModels().some(model => model.id === current.id && !model.damaged && isActive(model));
            const fromLocal = localTask();
            const validation = activating();
            const progressWords = state === 'ready' && active
                ? ['Model is ready and in use.', '模型已就绪，正在使用。'] :
                (validation && ACTIVATION_END[state]) || (fromLocal && LOCAL_END[state]) || PHASES[state];
            const done = current.done_bytes ?? current.done;
            const total = current.total_bytes ?? current.total;
            const measured = Number.isFinite(done) && Number.isFinite(total) && total > 0;
            progress.hidden = !progressWords;
            pause.hidden = !busy();
            resume.hidden = !PAUSED.has(state) || !(current.selection_id ?? current.id);
            progressActions.hidden = pause.hidden && resume.hidden;
            setText(pause, validation ? t('Cancel verification', '取消验证') : fromLocal ? t('Cancel import', '取消导入') : t('Pause', '暂停'));
            setText(resume, validation ? t('Retry verification', '重试验证') : fromLocal ? t('Retry copy', '重试复制') : t('Resume download', '继续下载'));
            meter.hidden = !busy();
            const transferring = TRANSFERRING.has(state) && measured;
            meter.set(transferring ? done / total : null);
            const amount = transferring ? `${bytesText(done)} / ${bytesText(total)}` : '';
            const detail = current.error ? userMessage(current.error, t, '') : '';
            const scenario = state === 'smoke' && (VALIDATION[current.validation] ? t(...VALIDATION[current.validation]) : current.validation);
            const text = [progressWords && t(...progressWords), scenario, amount, detail].filter(Boolean).join(' · ');
            if (status.textContent !== text) status.textContent = text;
            status.className = `st-slot${state === 'failed' ? ' is-warn' : ''}`;
            if (state === 'ready' && candidate?.id === current.id) showCandidate(null);
            if (validation && state === 'ready' && active) {
                if (announcedActivation !== current.id) {
                    announcedActivation = current.id;
                    onActivate?.(view.active);
                }
            } else announcedActivation = undefined;
        }
        controls();
    }

    function applyChange(result) {
        const value = result[words.response];
        if (value) update({local_available: view.local_available, comfy_h3_available: view.comfy_h3_available, ...value});
        refresh();
    }

    function activate(model) {
        return run('activate', {id: model.id, ...modelSelection(model)}, t(...words.switching), result => {
            applyChange(result);
            if (!activating()) onActivate?.(view.active);
        });
    }

    function reviewSelection(model, trigger = input) {
        showCandidate(model);
        reviewTrigger = trigger;
        candidateTitle.tabIndex = -1;
        if (candidate && review.isConnected) candidateTitle.focus();
    }
    const check = reference => {
        showCandidate(null);
        return run('inspect', {reference, ...selection()}, t('Checking model compatibility…', '正在检查模型兼容性…'),
            result => reviewSelection(result.candidate));
    };
    const pickLocal = (payload, trigger) => {
        showCandidate(null);
        return run('local-pick', payload,
            t('Waiting for the Windows file picker…', '请在 Windows 文件选择窗口中操作…'),
            result => reviewSelection(result.candidate, trigger));
    };
    localFile.onclick = () => pickLocal(family ? selection() : {selection: 'cache'}, localFile);
    if (localFolder) localFolder.onclick = () => pickLocal({selection: 'source'}, localFolder);
    if (comfyFolder) comfyFolder.onclick = () => run('local-pick', {selection: 'comfy_h3'},
        t('Choose a ComfyUI H3 weight folder in Windows…', '请在 Windows 中选择 ComfyUI H3 权重目录…'), result => {
            if (!Array.isArray(result.candidates)) return;
            showCandidate(null);
            applyChange(result);
            showComfyScan(result.candidates);
        });

    function comfySelection(entry) {
        const preset = entry.model.presets?.find(option => option.id === entry.select.value);
        const id = entry.checked.get(preset?.id) || preset?.installed_id;
        const installed = id ? view.installed?.find(model => model.id === id) : null;
        const ready = entry.model.compatible === true && Boolean(preset) && preset.compatible !== false;
        return {preset, installed, ready, pending: Boolean(id && id === task().id && busy())};
    }

    function paintComfyRow(entry, locked) {
        const {preset, installed, ready, pending} = comfySelection(entry);
        const active = installed && isActive(installed);
        const said = active ? t('In use', '使用中') : installed ? t('Installed', '已安装')
            : pending ? t('Importing', '正在导入') : ready ? t('Ready to check', '可检查兼容性')
                : preset?.compatibility_reason || entry.model.compatibility_reason || t('Unavailable', '不可用');
        if (entry.status.textContent !== said) entry.status.textContent = said;
        entry.status.className = active || installed ? 'is-ok' : ready ? '' : 'is-warn';
        const label = installed ? t('Use model', '使用模型') : t('Check', '检查');
        setText(entry.action, label);
        entry.select.disabled = locked || !entry.select.value || entry.model.compatible !== true;
        entry.action.disabled = locked || !(installed || ready);
        entry.action.hidden = Boolean(active || (!installed && !ready));
    }

    function showComfyScan(models) {
        comfyRows = models.map(model => {
            const title = model.label || model.name;
            const format = [formatName(model.format), Number.isFinite(model.bytes) && bytesText(model.bytes)].filter(Boolean).join(' · ');
            const select = el('select');
            select.setAttribute('aria-label', t(`Preset for ${title}`, `${title} 的预设`));
            for (const option of model.presets || []) {
                const node = el('option', '', option.label || option.id.toUpperCase());
                node.value = option.id;
                select.append(node);
            }
            const action = button(t('Check', '检查'), {kind: 'fill'});
            const status = el('span');
            const item = row({title, sub: format, end: [...(select.childElementCount ? [field(select)] : []), action]});
            item.sub.after(status);
            const entry = {model, select, action, status, checked: new Map(), element: item.element};
            select.onchange = () => {
                if (reviewTrigger === action) showCandidate(null);
                controls();
            };
            action.onclick = () => {
                const {preset, installed, ready} = comfySelection(entry);
                if (installed) return activate(installed);
                if (!ready) return;
                showCandidate(null);
                return run('local-comfy-inspect', {id: model.id, preset: preset.id},
                    t('Checking ComfyUI H3 weight compatibility…', '正在检查 ComfyUI H3 权重兼容性…'), result => {
                        if (result.candidate?.id) entry.checked.set(preset.id, result.candidate.id);
                        applyChange(result);
                        reviewSelection(result.candidate, action);
                    });
            };
            return entry;
        });
        comfyResults.replaceChildren(...(comfyRows.length ? comfyRows.map(entry => entry.element) : [
            el('p', 'st-p', t('No H3 checkpoint files were found in this folder.', '这个目录里没有找到 H3 检查点文件。')),
        ]));
        controls();
    }
    search.onclick = () => run('search', {query: input.value.trim()}, t('Searching Hugging Face…', '正在搜索 Hugging Face…'), result => {
        const models = result.models || [];
        results.replaceChildren(...models.map(model => {
            const open = button(t('Check', '检查'), {kind: 'fill'});
            open.onclick = () => check(model.repo);
            return row({title: model.repo, sub: model.license, end: [open]}).element;
        }));
        results.hidden = !models.length;
        if (!models.length) message(t('No models found. Try another search or paste a repository link.', '未找到模型。请尝试其他关键词或粘贴仓库链接。'));
    });
    inspect.onclick = () => check(input.value.trim());
    source.onsubmit = event => {
        event.preventDefault();
        if (!search.disabled) (/^\S+\/\S+$/.test(input.value.trim()) ? inspect : search).click();
    };
    input.oninput = controls;
    function resetSearch() {
        results.hidden = true;
        results.replaceChildren();
        showCandidate(null);
        message('');
        controls();
    }
    if (family) {
        family.onchange = () => { paintComponents(); resetSearch(); };
        component.onchange = resetSearch;
    }
    accept.onchange = controls;
    decline.onclick = () => { showCandidate(null); reviewTrigger.focus(); };
    const start = (model, fromLocal = false, id = model.id) => run(fromLocal ? 'local-copy' : 'install', {id, accepted: true, ...modelSelection(model)},
        fromLocal ? t('Starting copy…', '正在开始复制…') : t('Starting download…', '正在开始下载…'), result => {
            showCandidate(null);
            applyChange(result);
        });
    install.onclick = () => { if (candidate && accept.checked) start(candidate); };
    copyModel.onclick = () => { if (localModel(candidate)) start(candidate, true); };
    pause.onclick = () => run('cancel', {}, activating() ? t('Cancelling verification…', '正在取消验证…') :
        localTask() ? t('Cancelling import…', '正在取消导入…') : t('Pausing download…', '正在暂停下载…'), applyChange);
    resume.onclick = () => activating() ? activate(task().candidate || task()) :
        start(task().candidate || task(), localTask(), task().selection_id ?? task().id);
    return {elements, update, busy, focusComponent};
}
