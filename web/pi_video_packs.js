const STEPS = Object.freeze({fast2: 2, fast4: 4});
export const PI_VIDEO_MODES = Object.freeze(Object.keys(STEPS));
export const isPiVideoModel = mode => Object.hasOwn(STEPS, mode);
export const piVideoSteps = mode => STEPS[mode];
export const piVideoName = (t, mode) => t('pi-H3 · {steps} steps', 'pi-H3 · {steps} 步', {steps: piVideoSteps(mode)});

export function piVideoUnavailableText(t, reason) {
    if (reason === 'selected_component_changed') return t('The selected pi-H3 files changed. Repair them or restore the default in Model management.', '所选 pi-H3 文件已变更。请在模型管理中修复或恢复默认版本。');
    if (reason === 'engine') return t('pi-H3 needs the FreeVideo INT8 video model.', 'pi-H3 需要 FreeVideo INT8 视频模型。');
    if (reason === 'mac' || reason === 'gpu') return t('pi-H3 needs an NVIDIA GPU with CUDA.', 'pi-H3 需要支持 CUDA 的 NVIDIA 显卡。');
    return t('pi-H3 is unavailable on this installation.', '当前安装无法使用 pi-H3。');
}

export function piVideoDownloadText(t, status) {
    if (status?.compatible === false) return piVideoUnavailableText(t, status.compatibility_reason);
    if (isPiVideoModel(status?.blocked_by)) {
        return t('Waiting for {name} to finish downloading.', '等待 {name} 下载完成。', {name: piVideoName(t, status.blocked_by)});
    }
    const reason = status?.install?.reason || status?.install?.state;
    const message = {
        busy: ['Another pi-H3 model is downloading. Try again when it finishes.', '另一个 pi-H3 模型正在下载，请在下载完成后重试。'],
        disk_space: ['There is not enough disk space for the pi-H3 download.', '磁盘空间不足，无法下载 pi-H3。'],
        integrity_failed: ['The pi-H3 download failed verification. Retry to continue.', 'pi-H3 下载校验失败，请重试。'],
        not_published: ['This pi-H3 download is currently unavailable.', '此 pi-H3 版本当前无法下载。'],
        offline: ['The download service could not be reached. Retry to continue.', '无法连接下载服务，请重试。'],
        interrupted: ['The pi-H3 download was interrupted. Resume to continue.', 'pi-H3 下载已中断，可以继续下载。'],
        cancelled: ['The pi-H3 download was paused.', 'pi-H3 下载已暂停。'],
    }[reason];
    return message ? t(...message) : t('Could not finish installing pi-H3. Retry to continue.', 'pi-H3 安装未完成，请重试。');
}

const running = value => value?.install?.state === 'running';

export function piVideoPackState(t, status) {
    if (!status) return {state: 'checking'};
    const present = status.present === true;
    if (status.compatible === false) return {state: 'unavailable', present, reason: piVideoUnavailableText(t, status.compatibility_reason)};
    if (!present && status.published === false) return {state: 'unavailable', reason: piVideoDownloadText(t, {install: {reason: 'not_published'}})};
    const install = status.install || {};
    if (running(status)) {
        const verifying = install.phase === 'verify';
        const total = (verifying ? install.verify_total : install.total) || 0;
        const done = (verifying ? install.verify_done : install.done) || 0;
        return {state: 'downloading', verifying, cancelling: install.cancelling === true, total, done,
            fraction: total > 0 ? Math.min(1, Math.max(0, done / total)) : null};
    }
    if (present) return {state: 'ready', present};
    const resumeBytes = status.resume_bytes || 0;
    const resume = resumeBytes > 0;
    if (status.blocked_by) return {state: 'blocked', reason: piVideoDownloadText(t, status)};
    if (install.state === 'failed' && install.reason !== 'interrupted')
        return {state: 'failed', resume, resumeBytes, total: status.bytes, reason: piVideoDownloadText(t, status)};
    const paused = install.state === 'cancelled' || install.reason === 'interrupted';
    return {state: 'missing', total: status.bytes, resume: resume || paused, resumeBytes, paused,
        reason: paused ? piVideoDownloadText(t, status) : ''};
}

// Every pack view in a document uses the same responses and only one poll. The installer keeps its job
// when a view closes. A Generate button can wait for readiness without owning the download.
export function createPiVideoPacks(fetchApi, {idleMs = 3000, runningMs = 1000} = {}) {
    const values = new Map(), reads = new Map(), actions = new Map(), revisions = new Map(), misses = new Map();
    const subscriptions = new Set();
    let timer = 0;

    function status(mode) {
        const value = values.get(mode);
        if (!value || value.present || running(value) || value.blocked_by) return value;
        const other = PI_VIDEO_MODES.find(id => id !== mode && (running(values.get(id)) || actions.get(id)?.operation === 'install'));
        return other ? {...value, blocked_by: other} : value;
    }

    function notify() {
        for (const reflect of subscriptions) reflect();
    }

    function record(mode, value) {
        if (JSON.stringify(values.get(mode)) !== JSON.stringify(value)) { values.set(mode, value); notify(); }
        return status(mode);
    }

    function schedule(delay = PI_VIDEO_MODES.some(mode => running(values.get(mode))) ? runningMs : idleMs) {
        clearTimeout(timer);
        if (!subscriptions.size) return;
        timer = setTimeout(async () => {
            await refresh();
            schedule();
        }, delay);
    }

    function subscribe(reflect) {
        subscriptions.add(reflect);
        schedule(0);
        return () => {
            subscriptions.delete(reflect);
            if (!subscriptions.size) clearTimeout(timer);
        };
    }

    async function call(mode, operation, revision) {
        const last = values.get(mode);
        const job = operation === 'cancel' && last?.install?.job_id;
        const payload = {action: operation, mode, ...(job ? {job_id: job} : {})};
        const options = operation ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)} : {};
        const latest = () => (revisions.get(mode) || 0) === revision;
        try {
            const response = await fetchApi(`/freevideo/fast-pack${operation ? '' : `?mode=${mode}`}`, {cache: 'no-store', ...options});
            const body = await response.json();
            if (!latest()) return status(mode);
            if (response.ok) {
                misses.delete(mode);
                return record(mode, body);
            }
            if (operation && body?.reason) {
                // A failed cancel request must not replace a running job with a local error.
                if (operation === 'cancel' && running(last)) return status(mode);
                return record(mode, {...last, ...body, install: {state: 'failed', reason: body.reason}});
            }
        } catch { /* A brief disconnect keeps the last transfer progress. */ }
        if (!latest()) return status(mode);
        const missed = (misses.get(mode) || 0) + 1;
        misses.set(mode, missed);
        if (running(last) && missed < 5) return status(mode);
        // A status read error is different from a server download failure. Preserve any verified
        // installation while it is offline and allow another status read to settle the uncertainty.
        return record(mode, {...last, install: {state: 'failed', reason: 'offline'}});
    }

    function read(mode) {
        if (actions.has(mode)) return actions.get(mode).promise;
        if (reads.has(mode)) return reads.get(mode);
        const promise = call(mode, null, revisions.get(mode) || 0).finally(() => {
            if (reads.get(mode) === promise) reads.delete(mode);
        });
        reads.set(mode, promise);
        return promise;
    }

    function refresh() {
        return Promise.all(PI_VIDEO_MODES.map(read));
    }

    function action(mode, operation) {
        if (actions.has(mode)) return actions.get(mode).promise;
        const revision = (revisions.get(mode) || 0) + 1;
        revisions.set(mode, revision);
        // Do not make a subsequent polling request join a read that started before the action.
        reads.delete(mode);
        const promise = call(mode, operation, revision).finally(() => {
            actions.delete(mode);
            notify();
            schedule(0);
        });
        actions.set(mode, {operation, promise});
        notify();
        return promise;
    }

    async function ensure(mode) {
        let value = status(mode) || await read(mode);
        if (value?.present && !running(value)) return true;
        if (value?.compatible === false || value?.published === false || value?.blocked_by) return false;
        if (!running(value)) value = await action(mode, 'install');
        if (!running(value)) return value?.present === true;
        return new Promise(resolve => {
            const release = subscribe(() => {
                const latest = status(mode);
                if (running(latest)) return;
                release(); resolve(latest?.present === true);
            });
        });
    }

    return {status, read, refresh, subscribe, ensure, action, pending: mode => actions.has(mode)};
}

let packs;
export function piVideoPacks(fetchApi) {
    packs ??= createPiVideoPacks(fetchApi);
    return packs;
}
