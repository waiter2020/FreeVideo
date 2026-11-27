import {bar, button, card, el, row, setText} from './settings_kit.js';
import {bytesText} from './settings_words.js';
import {PI_VIDEO_MODES, piVideoName, piVideoPackState, piVideoPacks} from './pi_video_packs.js';

export function createSettingsPiVideoPacks({fetchApi, t}) {
    const packs = piVideoPacks(fetchApi);
    const element = card();
    element.classList.add('st-pi-video-packs');
    element.setAttribute('role', 'group');
    element.setAttribute('aria-label', t('pi-H3 video models', 'pi-H3 视频模型'));
    const rows = PI_VIDEO_MODES.map(mode => {
        const item = row({iconName: 'film', title: piVideoName(t, mode), cls: 'st-mrow'});
        const size = el('span', 'st-sz');
        const status = el('span', 'st-chip');
        const action = button('', {kind: 'fill'});
        const meter = bar('st-mbar');
        meter.setAttribute('aria-label', piVideoName(t, mode));
        item.sub.setAttribute('aria-live', 'polite');
        item.end.append(size, status, action);
        item.element.append(meter);
        item.element.dataset.mode = mode;
        element.append(item.element);
        return {mode, item, size, status, action, meter};
    });

    function paint() {
        for (const {mode, item, size, status, action, meter} of rows) {
            const state = piVideoPackState(t, packs.status(mode));
            const busy = state.state === 'downloading';
            item.element.dataset.state = state.state;
            item.sub.hidden = false;
            item.sub.className = busy ? 'is-accent' : state.state === 'failed' ? 'is-warn' : '';
            size.hidden = !state.total || busy || !['missing', 'failed'].includes(state.state);
            size.textContent = size.hidden ? '' : t('About {size}', '约 {size}', {size: bytesText(state.total)});
            status.hidden = !state.present && state.state !== 'unavailable';
            status.className = state.present ? 'st-chip is-ok' : 'st-chip is-warn';
            status.textContent = state.present ? t('Installed', '已安装') : t('Unavailable', '不可用');
            const retained = state.resumeBytes ? t('Kept {size} already downloaded.', '已保留下载的 {size}。', {size: bytesText(state.resumeBytes)}) : '';
            let detail = '', actionLabel = '';
            switch (state.state) {
                case 'ready':
                    detail = t('Select this model in Create or Edit.', '可在创作或剪辑视图选择。');
                    break;
                case 'downloading':
                    if (state.cancelling) detail = t('Pausing…', '正在暂停…');
                    else if (state.verifying) detail = state.fraction === null ? t('Verifying model files…', '正在校验模型文件…')
                        : t('Verifying model files · {percent}%', '正在校验模型文件 · {percent}%', {percent: Math.floor(state.fraction * 100)});
                    else detail = state.total ? t('Downloading · {done} / {total}', '正在下载 · {done} / {total}',
                        {done: state.done ? bytesText(state.done) : '0', total: bytesText(state.total)}) : t('Starting download…', '正在开始下载…');
                    actionLabel = t('Pause', '暂停');
                    break;
                case 'missing':
                    detail = [state.reason || t('Not installed', '未安装'), retained].filter(Boolean).join(' ');
                    actionLabel = state.resume ? t('Resume', '继续下载') : t('Install', '安装');
                    break;
                case 'failed':
                    detail = [state.reason, retained].filter(Boolean).join(' ');
                    actionLabel = t('Retry', '重试');
                    break;
                case 'unavailable':
                case 'blocked':
                    detail = state.reason;
                    break;
                default:
                    detail = t('Checking files…', '正在检查文件…');
            }
            if (item.sub.textContent !== detail) item.sub.textContent = detail;
            action.hidden = !actionLabel;
            action.disabled = packs.pending(mode) || state.cancelling === true;
            setText(action, actionLabel);
            action.onclick = () => { void packs.action(mode, busy ? 'cancel' : 'install'); };
            meter.hidden = !busy;
            meter.set(busy ? state.fraction : null);
        }
    }
    const dispose = packs.subscribe(paint);
    paint();
    return {element, dispose};
}
