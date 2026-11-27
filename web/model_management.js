import {bar, button, card, checkify, el, field, groupHead, row, slot} from './settings_kit.js';
import {bytesText, failureText} from './settings_words.js';

const groups = {
    video: ['Video generation', '视频生成', 'film'],
    image: ['Image generation', '图像生成', 'img'],
    prompt: ['Prompt and vision', '提示词与视觉理解', 'auto'],
    visual: ['Visual tools', '视觉工具', 'layers'],
    vision: ['Image and video tools', '图像与视频工具', 'layers'],
    audio: ['Audio and speech', '音频与语音', 'layers'],
    other: ['Other models', '其他模型', 'cube'],
};
const phases = {
    preparing: ['Preparing files…', '正在准备文件…'],
    downloading: ['Downloading…', '正在下载…'],
    copying: ['Copying…', '正在复制…'],
    verifying: ['Checking files…', '正在校验文件…'],
    waiting_for_gpu: ['Waiting for the GPU…', '正在等待 GPU…'],
    waiting: ['Waiting for the current task…', '正在等待当前任务…'],
    smoke: ['Testing the model…', '正在试运行模型…'],
    testing: ['Testing the model…', '正在试运行模型…'],
    activating: ['Switching models…', '正在切换模型…'],
    installing: ['Installing…', '正在安装…'],
    repairing: ['Repairing…', '正在修复…'],
    removing: ['Removing…', '正在移除…'],
    cancelling: ['Stopping…', '正在停止…'],
    ready: ['Model operation complete.', '模型操作已完成。'],
    complete: ['Model operation complete.', '模型操作已完成。'],
    completed: ['Model operation complete.', '模型操作已完成。'],
    cancelled: ['Model operation cancelled.', '模型操作已取消。'],
    failed: ['Model operation failed.', '模型操作失败。'],
};
const finished = new Set(['idle', 'ready', 'complete', 'completed', 'success', 'cancelled', 'failed', 'error']);
const activeJob = job => Boolean(job?.phase && !finished.has(job.phase));
const variant = (model, id) => model.versions?.find(version => version.id === id);
const usesDefault = model => model.active_id === 'builtin' || (model.active_id != null && model.active_id === model.default?.id);

function input({placeholder = '', type = 'text'} = {}) {
    const node = el('input');
    node.type = type;
    node.placeholder = placeholder;
    node.autocomplete = 'off';
    node.spellcheck = false;
    return node;
}

export function createModelManagement({request, refresh, t, onManage}) {
    const element = el('section', 'st-model-management');
    const heading = groupHead(t('Model library', '模型库'));
    const search = input({type: 'search', placeholder: t('Search models', '搜索模型')});
    search.setAttribute('aria-label', search.placeholder);
    const searchRow = el('div', 'st-model-search');
    searchRow.append(field(search));
    const list = el('div', 'st-model-library');
    const notice = slot('st-model-job-text');
    const progress = bar();
    progress.setAttribute('aria-label', t('Model operation progress', '模型操作进度'));
    const stop = button(t('Cancel', '取消'), {iconName: 'stop'});
    const jobBody = el('div', 'st-sub st-model-job');
    jobBody.append(notice, stop, progress);
    const jobPanel = card(jobBody);
    jobPanel.hidden = true;
    element.append(heading, searchRow, jobPanel, list);
    element.hidden = true;

    let view = {}, setupBusy = false, requesting = false, requestError = '', selected = '';
    let review = null, consent = false, source = 'local', location = '', revision = '', repositoryPath = '';
    let listKey = '', controls = [], focusConsent = false, consentInput, stopping = false;
    const toggles = new Map();

    const enabled = (model, action) => model.actions?.includes(action) === true;
    const unavailable = () => !view.local_available || setupBusy || requesting || activeJob(view.job);
    const searchValues = model => [model.id, model.label, model.label_zh, model.format, model.reason, ...(model.consumers || []),
        model.default?.label, model.default?.source, ...(model.versions || []).flatMap(version => [version.label, version.source, version.repo])];
    const name = model => t(model.label, model.label_zh || model.label);

    function sayJob() {
        const job = view.job;
        const busy = activeJob(job);
        if (!busy) stopping = false;
        const phase = stopping && busy ? 'cancelling' : job?.phase;
        const message = requestError || (job?.error && failureText(job.error)) || (phase && phase !== 'idle'
            ? t(...(phases[phase] || ['Working…', '正在处理…'])) : '');
        const model = view.slots?.find(item => item.id === job?.slot);
        const bytes = busy && job.total_bytes > 0 ? `${job.done_bytes > 0 ? bytesText(job.done_bytes) : '0'} / ${bytesText(job.total_bytes)}` : '';
        const text = message && [requestError ? '' : model && name(model), message, bytes].filter(Boolean).join(' · ');
        if (notice.textContent !== text) notice.textContent = text;
        notice.classList.toggle('is-warn', Boolean(requestError || job?.error));
        jobPanel.hidden = !message;
        stop.hidden = !busy || job.cancellable === false;
        stop.disabled = requesting || !view.local_available || phase === 'cancelling';
        progress.hidden = !busy;
        if (busy) progress.set(job.total_bytes > 0 ? (job.done_bytes || 0) / job.total_bytes : null);
    }

    function syncControls() {
        for (const [node, disabled] of controls) node.disabled = unavailable() || disabled();
        sayJob();
    }

    function controlled(node, disabled = () => false) {
        controls.push([node, disabled]);
        return node;
    }

    function update(next, blocked = setupBusy) {
        setupBusy = blocked;
        if (next !== undefined) view = next || {};
        element.hidden = !view.slots?.length;
        paint();
    }

    async function run(action, id, payload = {}) {
        if (requesting) return;
        requesting = true;
        requestError = '';
        syncControls();
        try {
            const result = await request('model-component', {action, slot: id, ...payload});
            if (result.model_components) view = result.model_components;
            if (action === 'cancel') stopping = true;
            if (selected === id) {
                if (action === 'inspect') showReview({action: 'install', candidate: result.candidate});
                else if (action === 'pick' && result.path) { location = result.path; listKey = ''; }
                else if (['install', 'remove', 'repair'].includes(action)) showReview(null);
            }
            if (action !== 'inspect' && action !== 'pick') void refresh?.();
        } catch (error) {
            requestError = failureText(error?.message) || t('The model operation could not start.', '无法开始模型操作。');
        } finally {
            requesting = false;
            paint();
        }
    }
    stop.onclick = () => run('cancel', view.job.slot);

    function showReview(next) {
        review = next;
        consent = false;
        focusConsent = Boolean(next);
        listKey = '';
    }

    function choose(id) {
        selected = selected === id ? '' : id;
        source = 'local';
        location = revision = repositoryPath = '';
        showReview(null);
        paint();
        toggles.get(id)?.focus();
    }

    function versionRow(model, version, builtin = false) {
        const chosen = builtin ? usesDefault(model) : model.active_id === version.id;
        const installed = version.installed !== false;
        const commands = version.actions || model.actions || [];
        const origin = version.repo || (version.source === 'local' ? t('Local files', '本地文件') : version.source);
        const description = [builtin && t('Default', '默认'), origin, version.revision?.slice(0, 12),
            version.bytes > 0 && bytesText(version.bytes), version.damaged ? t('Needs repair', '需要修复') : !installed && t('Not installed', '未安装')];
        const item = row({title: version.label, sub: description.filter(Boolean).join(' · '), cls: 'st-model-version'});
        if (chosen && installed) item.end.append(el('span', 'st-chip is-ok', t('In use', '使用中')));
        const action = (operation, words, handler) => {
            if (!commands.includes(operation)) return;
            const control = controlled(button(t(...words), {kind: 'fill'}));
            control.onclick = handler || (() => run(operation, model.id, {version_id: builtin ? 'builtin' : version.id}));
            item.end.append(control);
        };
        if (!installed && builtin) action('install', ['Install', '安装'], () => {
            showReview({action: 'install', version}); paint();
        });
        if (builtin && !chosen && installed && enabled(model, 'reset')) {
            const reset = controlled(button(t('Restore default', '恢复默认'), {kind: 'fill'}));
            reset.onclick = () => run('reset', model.id);
            item.end.append(reset);
        }
        if (installed) {
            if (!chosen && !builtin) action('activate', ['Use model', '使用模型']);
            action('verify', ['Check files', '校验文件']);
            if (!chosen) action('remove', ['Remove', '移除'], () => { showReview({action: 'remove', version, builtin}); paint(); });
        }
        action('repair', ['Repair', '修复'], () => { showReview({action: 'repair', version, builtin}); paint(); });
        return item.element;
    }

    function labelled(text, control) {
        const label = el('label', 'st-label');
        label.append(el('span', '', text), field(control, {mono: control.tagName === 'INPUT'}));
        return label;
    }

    function importForm(model) {
        const sources = el('select');
        for (const [value, words] of [['local', ['Local files', '本地文件']], ['hf', ['Hugging Face', 'Hugging Face']]]) {
            if (model.sources && !model.sources.includes(value)) continue;
            const option = el('option', '', t(...words)); option.value = value; sources.append(option);
        }
        if (model.sources && !model.sources.includes(source)) source = model.sources[0] || 'local';
        sources.value = source;
        sources.onchange = () => { source = sources.value; location = ''; showReview(null); paint(); };
        const path = input({placeholder: source === 'hf' ? 'org/model'
            : t('Path in a configured model folder', '已配置模型文件夹中的路径')});
        path.value = location;
        const rev = input(), subdirectory = input();
        rev.value = revision;
        subdirectory.value = repositoryPath;
        const form = el('form', 'st-model-import');
        const actions = el('div', 'st-actions');
        form.append(labelled(t('Model source', '模型来源'), controlled(sources)),
            labelled(source === 'hf' ? t('Repository ID', '仓库 ID') : t('Local model file or folder', '本地模型文件或文件夹'), controlled(path)));
        if (source === 'hf') form.append(labelled(t('Revision (optional)', '版本（可选）'), controlled(rev)),
            labelled(t('Repository path (optional)', '仓库内路径（可选）'), controlled(subdirectory)));
        if (source === 'local' && (view.native_picker === true || enabled(model, 'pick'))) {
            const pick = controlled(button(t('Browse', '浏览'), {iconName: 'folder'}));
            pick.onclick = () => run('pick', model.id);
            actions.append(pick);
        }
        const inspect = controlled(button(t('Inspect model', '检查模型'), {kind: 'fill'}), () => !location.trim());
        inspect.type = 'submit';
        const change = () => {
            location = path.value; revision = rev.value; repositoryPath = subdirectory.value;
            if (review) { showReview(null); form.nextElementSibling?.remove(); }
            syncControls();
        };
        path.oninput = rev.oninput = subdirectory.oninput = change;
        form.onsubmit = event => {
            event.preventDefault();
            if (unavailable() || !location.trim()) return;
            const payload = source === 'local' ? {source, path: location.trim()} : {source, repo: location.trim()};
            if (source === 'hf' && revision.trim()) payload.revision = revision.trim();
            if (source === 'hf' && repositoryPath.trim()) payload.path = repositoryPath.trim();
            return run('inspect', model.id, payload);
        };
        actions.append(inspect); form.append(actions);
        return form;
    }

    function reviewPanel(model) {
        const {action, candidate, version, builtin} = review;
        const item = candidate || version;
        const panel = el('div', 'st-model-review');
        panel.append(el('strong', '', item.label));
        const files = Array.isArray(item.files) ? item.files.length : item.files;
        const bytes = item.download_bytes || item.bytes;
        const meta = [item.repo, item.revision, bytes > 0 && bytesText(bytes),
            files > 0 && t('{count} files', '{count} 个文件', {count: files}), item.license];
        if (meta.some(Boolean)) panel.append(el('p', 'st-p', meta.filter(Boolean).join(' · ')));
        if (item.download_scope) panel.append(el('p', 'st-p', t('Includes the {name} pack.', '包含 {name} 模型包。', {name: item.download_scope})));
        const repository = item.repo || item.source || '';
        if (/^[\w.-]+\/[\w.-]+$/.test(repository)) {
            const modelCard = el('a', '', t('View model card', '查看模型卡片'));
            modelCard.href = `https://huggingface.co/${repository}${item.revision ? '/tree/' + encodeURIComponent(item.revision) : ''}`;
            modelCard.target = '_blank';
            modelCard.rel = 'noopener noreferrer';
            panel.append(modelCard);
        }
        const confirming = action === 'remove'
            ? t('Remove this installed version?', '移除此已安装版本？')
            : action === 'repair' ? t('Repair this installed version?', '修复此已安装版本？')
                : t('I agree to install this model and accept its license.', '我同意安装此模型并接受其许可证。');
        const approve = checkify(input({type: 'checkbox'})); approve.checked = consent;
        consentInput = approve;
        approve.onchange = () => { consent = approve.checked; syncControls(); };
        const check = el('label', 'st-agree'); check.append(controlled(approve), el('span', '', confirming));
        const done = controlled(button(action === 'remove' ? t('Remove version', '移除版本')
            : action === 'repair' ? t('Repair version', '修复版本') : t('Install version', '安装版本'), {kind: 'primary'}), () => !consent);
        const back = button(t('Back', '返回'));
        back.onclick = () => { showReview(null); paint(); };
        done.onclick = () => {
            const payload = candidate ? {candidate_id: candidate.id} : {version_id: builtin || action === 'install' ? 'builtin' : version.id};
            return run(action, model.id, {...payload, accepted: true});
        };
        const actions = el('div', 'st-actions'); actions.append(done, back);
        panel.append(check, actions);
        return panel;
    }

    function modelRow(model, iconName) {
        const active = usesDefault(model) ? model.default : variant(model, model.active_id);
        const installed = Boolean(model.supported !== false && active && active.installed !== false);
        const label = model.supported === false ? t('Unavailable', '不可用')
            : installed ? t('In use', '使用中') : active?.damaged ? t('Needs repair', '需要修复') : t('Not installed', '未安装');
        const chip = el('span', installed ? 'st-chip is-ok' : active?.damaged ? 'st-chip is-warn' : 'st-chip', label);
        const managed = model.manager && onManage && !model.actions?.length;
        const expanded = selected === model.id;
        const accessLabel = managed ? t('Open controls', '打开管理入口') : expanded ? t('Close', '收起') : t('Manage', '管理');
        const access = button(accessLabel, {kind: 'fill'});
        access.dataset.modelToggle = model.id;
        access.setAttribute('aria-label', `${accessLabel} · ${name(model)}`);
        toggles.set(model.id, access);
        if (!managed) access.setAttribute('aria-expanded', String(expanded));
        access.onclick = managed ? () => onManage(model) : () => choose(model.id);
        const description = [active?.label || model.default?.label, model.format, model.reason].filter(Boolean).join(' · ');
        const entry = row({iconName, title: name(model), sub: description, end: [chip, access]});
        if (!expanded) return [entry.element];
        const detail = el('div', 'st-sub st-model-detail');
        if (model.consumers?.length) detail.append(el('p', 'st-p', t('Used by {names}', '用于 {names}', {names: model.consumers.join(' · ')})));
        if (model.default) detail.append(versionRow(model, model.default, true));
        for (const version of model.versions || []) detail.append(versionRow(model, version));
        if (model.manager && onManage) {
            const open = button(t('Open import controls', '打开导入入口'), {kind: 'fill'});
            open.onclick = () => onManage(model);
            detail.append(open);
        }
        if (model.supported !== false && enabled(model, 'install')) {
            if (view.local_available) detail.append(importForm(model));
            else detail.append(el('p', 'st-p', t('Open FreeVideo on the host computer to change models.', '在运行 FreeVideo 的电脑上打开设置即可更换模型。')));
        }
        if (review?.candidate || review?.version) detail.append(reviewPanel(model));
        return [entry.element, detail];
    }

    function paint() {
        const key = JSON.stringify([view.slots, view.local_available, view.native_picker, search.value, selected]);
        if (key !== listKey) {
            listKey = key;
            controls = [];
            consentInput = undefined;
            toggles.clear();
            const query = (search.value || '').trim().toLowerCase();
            const sections = new Map();
            for (const model of view.slots || []) {
                if (query && !searchValues(model).some(value => String(value || '').toLowerCase().includes(query))) continue;
                const group = Object.hasOwn(groups, model.group) ? model.group : 'other';
                if (!sections.has(group)) sections.set(group, card());
                sections.get(group).append(...modelRow(model, groups[group][2]));
            }
            list.replaceChildren(...Object.entries(groups).flatMap(([group, [en, zh]]) => sections.has(group)
                ? [groupHead(t(en, zh)), sections.get(group)] : []));
            if (!list.childElementCount) list.append(el('p', 'st-p', t('No models match this search.', '没有找到符合条件的模型。')));
        }
        syncControls();
        if (focusConsent && consentInput) { consentInput.focus(); focusConsent = false; }
    }

    search.oninput = paint;
    return {element, update, busy: () => activeJob(view.job)};
}
