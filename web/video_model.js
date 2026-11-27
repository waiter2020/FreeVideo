// An installation set up with MiniMax H3 Image alone has no video model. The video panel's main button then adds it:
// the launcher reading this ComfyUI opens its installation with the video model, and its review installs it once accepted.
import {modelPicker} from './image_view_models.js';
import {isPiVideoModel, PI_VIDEO_MODES, piVideoName, piVideoPackState, piVideoPacks, piVideoSteps,
    piVideoUnavailableText} from './pi_video_packs.js';
export {isPiVideoModel, piVideoDownloadText, piVideoSteps} from './pi_video_packs.js';

export const VIDEO_MODEL_CHANGED = 'freevideo-video-model-changed';
export const COMFY_H3_SAMPLING = Object.freeze({base_steps: 20, two_pass: false, preview: false, step_cache: 'none',
    upscale_factor: '2', fast_mode: 'none'});
const fastRow = (grid, mode) => grid?.fast_modes?.[mode] ?? (mode === 'fast2' ? grid?.fast_mode : undefined);

export function piVideoModeInfo(grid, mode, width, height, frames) {
    const row = fastRow(grid, mode), known = grid?.status === 'ready';
    const i = grid?.sizes?.findIndex(size => size.width === Number(width) && size.height === Number(height)) ?? -1;
    const j = grid?.lengths?.findIndex(length => length.frames === frames) ?? -1;
    return {known, listed: row?.listed === true, reason: row?.unlisted_reason,
        maxFrames: row?.max_frames?.[i], fits: typeof row?.plan?.[i]?.[j] === 'number' ? row.plan[i][j] >= 0 : undefined,
        seconds: row?.estimate_seconds?.[i]?.[j]};
}

export function piVideoProblem(t, mode, {continuation, multipart, originalAudio, preview, settingsLinked, info, frames} = {}) {
    if (!isPiVideoModel(mode)) return '';
    if (continuation || multipart) return t('pi-H3 generates one independent clip. Choose a single clip or switch models.', 'pi-H3 每次生成一段独立视频。请选择单段，或切换模型。');
    if (originalAudio) return t('pi-H3 generates its own audio. Change the audio mode or switch models.', 'pi-H3 会生成自己的音频。请更改音频用法，或切换模型。');
    if (preview) return t('pi-H3 generates a finished clip in one pass. Turn off preview or switch models.', 'pi-H3 单次采样生成成片。请关闭预览，或切换模型。');
    if (settingsLinked) return t('pi-H3 uses fixed sampling settings. Disconnect linked sampling inputs in Nodes view.', 'pi-H3 使用固定采样设置。请在节点视图断开采样输入的连线。');
    if (info?.known && !info.listed) return piVideoUnavailableText(t, info.reason);
    if (info?.fits === false || info?.maxFrames === null || Number.isInteger(info?.maxFrames) && frames > info.maxFrames)
        return t('This length and size exceed the device limit for pi-H3. Shorten the clip or reduce its size.', '此时长和尺寸超出设备的 pi-H3 生成上限。请缩短视频或减小尺寸。');
    return '';
}

export async function activeVideoModel(fetchApi) {
    try {
        const response = await fetchApi('/freevideo/setup', {cache: 'no-store'});
        if (response.ok) return (await response.json())?.video_models?.active;
    } catch { /* Keep the last known profile while the engine is unavailable. */ }
}

export function applyVideoModelSnapshot(prompt, nodeId, model, selection = null) {
    const entry = prompt?.output?.[String(nodeId)];
    if (entry?.class_type !== 'FreeVideoGenerate' || !entry.inputs) return prompt;
    if (isPiVideoModel(selection)) {
        const fixed = {...COMFY_H3_SAMPLING, base_steps: piVideoSteps(selection), refine_steps: 3, fast_mode: selection};
        if (Object.keys(fixed).some(name => Array.isArray(entry.inputs[name]))) throw new Error('pi_h3_linked_sampling');
        Object.assign(entry.inputs, fixed, {native_h3_preset: ''});
        return prompt;
    }
    if (selection === 'current' && !Array.isArray(entry.inputs.fast_mode)) entry.inputs.fast_mode = 'none';
    if (model === undefined) return prompt;
    const native = model?.kind === 'comfy_h3';
    entry.inputs.native_h3_preset = native ? model.id : '';
    if (!native) return prompt;
    for (const [name, setting] of Object.entries(COMFY_H3_SAMPLING)) {
        if (!Array.isArray(entry.inputs[name])) entry.inputs[name] = setting;
    }
    return prompt;
}

export function createVideoModelPicker({fetchApi, t, scope, initial = null, onChange = () => {}, onActive = () => {}}) {
    const css = new URL('./image_view.css', import.meta.url).href;
    if (![...document.querySelectorAll('link[rel=stylesheet]')].some(link => link.href === css)) {
        const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = css; document.head.append(link);
    }
    const key = `freevideo.video.model.${scope}`, packs = piVideoPacks(fetchApi);
    let selected = initial, active, grid, disposed = false;
    try {
        const saved = localStorage.getItem(key);
        if (saved === 'current' || isPiVideoModel(saved)) selected = saved;
    } catch { /* The choice still works for this view. */ }
    const models = [{id: 'current', name: t('Native H3', '原生H3'), short: t('Model', '模型')},
        ...PI_VIDEO_MODES.map(id => ({id, name: piVideoName(t, id), short: 'pi-H3'}))];
    const stateOf = (_, mode) => {
        if (mode === 'current') return {state: 'ready'};
        const row = fastRow(grid, mode), state = piVideoPackState(t, packs.status(mode));
        if (state.state !== 'unavailable' && grid?.status === 'ready' && !row?.listed)
            return {state: 'unavailable', reason: piVideoUnavailableText(t, row?.unlisted_reason)};
        return {...state, pending: packs.pending(mode)};
    };
    function statusLabel(state) {
        if (state.reason) return state.reason;
        if (state.state === 'ready') return t('Ready', '就绪');
        if (state.state === 'missing') return state.total
            ? t('Download · {size} GB', '需下载 · {size} GB', {size: (state.total / 1e9).toFixed(2)}) : t('Download needed', '需下载');
        if (state.state === 'downloading') {
            if (state.cancelling) return t('Stopping…', '正在停止…');
            if (state.verifying) return state.total ? t('Verifying {percent}%', '正在校验 {percent}%', {percent: Math.floor(state.fraction * 100)}) : t('Verifying…', '正在校验…');
            return state.total ? `${Math.floor(state.fraction * 100)}%` : t('Starting…', '正在开始…');
        }
        return t('Checking…', '正在检查…');
    }
    const picker = modelPicker({models, readState: stateOf, menuClass: 'fv-video-model-menu', host: () => element.closest('dialog') || element.closest('.fv-studio, .fv-editor'),
        describe: (model, state) => state.reason || (model.id === 'current'
            ? active?.label || t('Use the active video model from Settings.', '使用设置中启用的视频模型。')
            : t('Generate a video with synchronized audio in {steps} steps.', '用 {steps} 步生成带同步音频的视频。', {steps: piVideoSteps(model.id)})),
        labelStatus: statusLabel,
        onPick: model => select(model.id),
        onInstall: model => { void ensure(model.id); },
        onPause: model => { void packs.action(model.id, 'cancel'); },
        actionLabel: state => state.pending || state.cancelling ? '' : ({missing: state.resume ? t('Resume', '继续') : t('Install', '安装'),
            failed: t('Retry', '重试'), downloading: t('Pause', '暂停')})[state.state] || '',
    });
    const element = document.createElement('div'); element.className = 'fv-video-model-picker';
    picker.element.dataset.minMenuWidth = '242'; element.append(picker.element);
    picker.element.querySelector('.fv-iv-mpick-status').setAttribute('aria-live', 'polite');
    const paint = () => { if (!disposed) picker.paint(null, selected || 'current', false); };
    function select(mode) {
        if (selected === mode || (mode !== 'current' && !isPiVideoModel(mode))) return;
        selected = mode;
        try { localStorage.setItem(key, selected); } catch { /* The view still retains the choice. */ }
        paint(); onChange();
        if (isPiVideoModel(mode)) void packs.read(mode);
    }
    async function ensure(mode, isCurrent = () => true) {
        if (!isPiVideoModel(mode) || disposed || stateOf(null, mode).state === 'unavailable') return false;
        return await packs.ensure(mode) && !disposed && isCurrent();
    }
    const changed = event => setActive(event.detail);
    function setActive(model) { if (model !== undefined) { active = model; paint(); onActive(model); } }
    window.addEventListener(VIDEO_MODEL_CHANGED, changed);
    const unsubscribe = packs.subscribe(paint);
    void activeVideoModel(fetchApi).then(model => { if (active === undefined && !disposed) setActive(model); });
    paint();
    return {element, selection: () => selected, select, active: () => active, setActive,
        setCapabilities(next) { if (next?.status === 'ready') { grid = next; paint(); } },
        info: (mode, width, height, frames) => {
            const status = packs.status(mode);
            if (status?.compatible === false) return {known: true, listed: false, reason: status.compatibility_reason};
            return piVideoModeInfo(grid, mode, width, height, frames);
        },
        ensure, status: packs.status, close: () => picker.close(false),
        dispose() {
            disposed = true; unsubscribe(); picker.dispose();
            window.removeEventListener(VIDEO_MODEL_CHANGED, changed);
        }};
}

// {installed, launcher}; a check that fails counts as installed, so the panel stays as it always was.
export async function videoModelState(fetchApi) {
    try {
        const response = await fetchApi('/freevideo/launcher/models', {cache: 'no-store'});
        if (!response.ok) return {installed: true, launcher: false};
        const value = await response.json();
        return {installed: value?.video !== false, launcher: value?.launcher === true};
    } catch { return {installed: true, launcher: false}; }
}

// The line shown after the click: where the user continues.
export async function addVideoModel(fetchApi, t) {
    let launcher = false;
    try {
        const response = await fetchApi('/freevideo/launcher/add-video', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        launcher = response.ok && (await response.json())?.launcher === true;
    } catch { launcher = false; }
    return launcher
        ? t('Confirm the download in the launcher.', '请在启动器中确认下载。')
        : t('Open the FreeVideo launcher to add the video model.', '请打开 FreeVideo 启动器添加视频模型。');
}

export const addVideoLabel = t => t('Add video model', '添加视频模型');
export const missingVideoText = t => t('Video model not installed.', '视频模型未安装。');
