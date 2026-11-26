// An installation set up with MiniMax H3 Image alone has no video model. The video panel's main button then adds it:
// the launcher reading this ComfyUI opens its installation with the video model, and its review installs it once accepted.

export const VIDEO_MODEL_CHANGED = 'freevideo-video-model-changed';
export const COMFY_H3_SAMPLING = Object.freeze({base_steps: 20, two_pass: false, preview: false, step_cache: 'none', upscale_factor: '2'});

export async function activeVideoModel(fetchApi) {
    try {
        const response = await fetchApi('/freevideo/setup', {cache: 'no-store'});
        if (response.ok) return (await response.json())?.video_models?.active;
    } catch { /* Keep the last known profile while the engine is unavailable. */ }
}

export function applyVideoModelSnapshot(prompt, nodeId, model) {
    const entry = prompt?.output?.[String(nodeId)];
    if (model === undefined || entry?.class_type !== 'FreeVideoGenerate' || !entry.inputs) return prompt;
    const native = model?.kind === 'comfy_h3';
    entry.inputs.native_h3_preset = native ? model.id : '';
    if (!native) return prompt;
    for (const [name, setting] of Object.entries(COMFY_H3_SAMPLING)) {
        if (!Array.isArray(entry.inputs[name])) entry.inputs[name] = setting;
    }
    return prompt;
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
