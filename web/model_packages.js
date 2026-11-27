// A package binds verified weights from the existing model libraries.
import {button, card, el, field, groupHead, row, slot} from './settings_kit.js';
import {IMAGE_COMPONENTS, IMAGE_FORMATS} from './settings_video_models.js';
import {IMAGE_MODEL_CHANGED} from './image_view_models.js';
import {MODEL_PACKAGES_CHANGED} from './video_model.js';

const roleName = (role, t) => t(...(IMAGE_COMPONENTS[role.id] || [role.label, role.label_zh]));
const choiceLabel = (choice, t) => choice.builtin ? t(choice.label || 'Built-in', choice.label_zh || '内置') : (choice.label || String(choice.id || '').slice(0, 8));

function labelled(text, control) {
    const label = el('label', 'st-label');
    label.append(el('span', '', text), field(control));
    return label;
}

function nameField(t) {
    const input = el('input');
    input.type = 'text';
    input.maxLength = 80;
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.setAttribute('aria-label', t('Package name', '模型包名称'));
    return input;
}

function roleSelect(role, value, t, disabled = false) {
    const select = el('select');
    select.disabled = disabled;
    select.setAttribute('aria-label', roleName(role, t));
    const choices = [...(role.choices || [])];
    if (!choices.some(choice => choice.id === value)) choices.push({id: value});
    for (const choice of choices) {
        const option = el('option', '', choiceLabel(choice, t));
        option.value = choice.id;
        select.append(option);
    }
    select.value = value;
    return select;
}

export function createModelPackages({request, refresh, onImportImage, t}) {
    const element = el('section', 'st-model-packages');
    const status = slot(), verification = slot();
    const cancel = button(t('Cancel verification', '取消验证'), {kind: 'fill'});
    const jobRow = card(verification, cancel); jobRow.hidden = true;
    const list = el('div');
    element.append(status, jobRow, list);

    const drafts = new Map(), fields = new Map();
    let doc, key = '', working = false, setupBusy = false, actions = [];
    const job = () => doc?.job;
    const verifying = () => ['waiting_for_gpu', 'smoke', 'cancelling'].includes(job()?.phase);

    function syncControls() {
        for (const action of actions) action.disabled = working || setupBusy || verifying() || doc?.local_available === false;
        cancel.disabled = working || job()?.phase === 'cancelling' || doc?.local_available === false;
    }

    function action(text, kind, onclick) {
        const item = button(text, {kind});
        item.onclick = onclick;
        actions.push(item);
        return item;
    }

    function track(control, id, value, event = 'oninput') {
        control.value = drafts.get(id) ?? value;
        control.dataset.packageField = id;
        control[event] = () => { if (control.value === value) drafts.delete(id); else drafts.set(id, control.value); };
        fields.set(id, control);
        return control;
    }

    function say(text, kind = '') {
        status.textContent = text;
        status.className = 'st-slot' + (kind ? ' is-' + kind : '');
    }

    function changed() {
        window.dispatchEvent(new Event(MODEL_PACKAGES_CHANGED));
        window.dispatchEvent(new Event(IMAGE_MODEL_CHANGED));
    }

    async function run(task) {
        if (working || doc?.local_available === false) return;
        working = true; syncControls();
        try { await task(); }
        finally { working = false; syncControls(); }
    }

    async function act(payload, saved = []) {
        if (setupBusy || verifying()) return;
        await run(async () => {
            try {
                const result = (await request('model-package', payload))?.model_packages;
                const waiting = payload.action === 'set-shared-encoder' && result?.job;
                if (!waiting) for (const [id, value] of saved) if (drafts.get(id) === value) drafts.delete(id);
                if (result) update({local_available: doc?.local_available, ...result});
                say(waiting ? '' : t('Saved.', '已保存。'), waiting ? '' : 'ok');
                if (!waiting) changed();
                void refresh?.();
            } catch (error) {
                say(error?.message || t('The model package could not be changed.', '模型包没能改好。'), 'warn');
            }
        });
    }

    function manageWeights(family, role) {
        if (!onImportImage?.(family, role)) say(t('Install the built-in component before importing weights.', '请先安装内置组件，再导入权重。'), 'warn');
    }

    function packageCard(spec, pack) {
        const builtin = pack.builtin === true, chips = [], isImage = spec.category === 'image';
        const active = isImage && doc.active?.image === pack.id;
        if (active) chips.push(el('span', 'st-chip is-ok', t('Current', '当前')));
        if (!isImage && doc.legacy_fast?.[spec.mode] === pack.id) chips.push(el('span', 'st-chip', t('Fast default', '快档默认')));
        if (pack.ready === false) chips.push(el('span', 'st-chip is-warn', t('Unavailable', '不可用')));
        const parameters = (spec.parameters || []).map(item => {
            const value = pack.parameters?.[item.key] ?? item.value;
            return t(item.label, item.label_zh) + ' ' + (Array.isArray(value) ? value.join(', ') : value);
        }).join(' · ');
        const title = builtin ? t(spec.label, spec.label_zh) : pack.name;
        const head = row({iconName: isImage ? 'img' : 'film', title, sub: parameters, end: chips});
        const body = el('div', 'st-sub'), buttons = el('div', 'st-actions');
        const nameKey = pack.id + '/name', selects = new Map();
        const name = builtin ? null : track(nameField(t), nameKey, pack.name || '');
        if (name) body.append(labelled(t('Name', '名称'), name));

        for (const role of spec.roles || []) {
            const id = pack.id + '/' + role.id, value = pack.bindings?.[role.id] || 'builtin';
            const select = roleSelect(role, drafts.get(id) ?? value, t, builtin);
            if (!builtin) selects.set(role.id, track(select, id, value, 'onchange'));
            const format = IMAGE_FORMATS[spec.family]?.[role.id] || [role.format || '', role.format_zh || ''];
            const block = el('div');
            block.append(labelled(roleName(role, t), select));
            if (format[0]) block.append(el('p', 'st-p', t(...format)));
            if (!builtin && isImage) block.append(action(t('Manage weights', '管理权重'), 'fill', () => manageWeights(spec.family, role.id)));
            body.append(block);
        }

        if (!builtin) {
            buttons.append(action(t('Save', '保存'), 'primary', () => {
                const bindings = Object.fromEntries([...selects].filter(([role, select]) => select.value !== pack.bindings?.[role])
                    .map(([role, select]) => [role, select.value]));
                const payload = {action: 'update', id: pack.id, name: name.value, bindings};
                return act(payload, [[nameKey, name.value], ...[...selects].map(([role, select]) => [pack.id + '/' + role, select.value])]);
            }));
            if (!active) buttons.append(action(t('Remove', '移除'), 'fill', () => act({action: 'remove', id: pack.id},
                [...drafts].filter(([id]) => id.startsWith(pack.id + '/')))));
        }
        if (isImage && !active) buttons.append(action(t('Use for pictures', '用于出图'), 'fill',
            () => act({action: 'activate', id: pack.id, surface: 'image'})));
        if (buttons.childElementCount) body.append(buttons);
        return card(head.element, body);
    }

    function addForm(spec) {
        const id = 'create/' + spec.id;
        const name = track(nameField(t), id, '');
        name.placeholder = t(spec.label, spec.label_zh);
        const add = action(t('Add package', '添加模型包'), 'fill', () => {
            const payload = {action: 'create', type: spec.id}, typed = name.value.trim();
            if (typed) payload.name = typed;
            return act(payload, [[id, name.value]]);
        });
        const body = el('div', 'st-sub'), buttons = el('div', 'st-actions');
        buttons.append(add); body.append(labelled(t('Name', '名称'), name), buttons);
        return card(body);
    }

    function sharedEncoder() {
        const spec = doc.types?.find(type => type.id === 'minimax-h3-image');
        const role = spec?.roles?.find(item => item.id === 'text_encoder');
        if (!role) return [];
        const value = doc.shared_encoder || 'builtin', id = 'shared/text_encoder';
        const select = track(roleSelect(role, drafts.get(id) ?? value, t), id, value, 'onchange');
        const save = action(t('Save', '保存'), 'primary', () => act({action: 'set-shared-encoder', id: select.value}, [[id, select.value]]));
        const bring = action(t('Manage weights', '管理权重'), 'fill', () => manageWeights(spec.family, role.id));
        const buttons = el('div', 'st-actions'), body = el('div', 'st-sub');
        buttons.append(save, bring);
        body.append(labelled(roleName(role, t), select), buttons);
        return [groupHead(t('Shared video text encoder', '视频共用文本编码器')), card(body)];
    }

    function section(category, title) {
        const nodes = [groupHead(t(...title))];
        for (const spec of (doc.types || []).filter(type => type.category === category)) {
            nodes.push(el('p', 'st-p', t(spec.label, spec.label_zh)));
            for (const pack of (doc.packages || []).filter(item => item.type === spec.id)) nodes.push(packageCard(spec, pack));
            nodes.push(addForm(spec));
        }
        return nodes;
    }

    function paint() {
        if (doc === undefined) return;
        if (!doc) {
            if (key !== 'null') {
                key = 'null'; actions = []; fields.clear();
                list.replaceChildren(el('p', 'st-p', t('Model packages are unavailable right now.', '模型包暂时不可用。')));
            }
            return;
        }
        const next = JSON.stringify([doc.types, doc.packages, doc.active, doc.legacy_fast, doc.shared_encoder]);
        if (next === key) { syncControls(); return; }
        key = next;
        const focused = document.activeElement, activeId = focused?.dataset?.packageField;
        const restoreFocus = activeId && fields.get(activeId) === focused;
        actions = []; fields.clear();
        list.replaceChildren(...section('video', ['pi-H3 packages', 'pi-H3 模型包']),
            ...sharedEncoder(), ...section('image', ['Image packages', '图像模型包']));
        syncControls();
        const replacement = restoreFocus && fields.get(activeId);
        if (replacement) {
            replacement.focus({preventScroll: true});
            if (replacement.type === 'text' && typeof focused.selectionStart === 'number')
                replacement.setSelectionRange?.(focused.selectionStart, focused.selectionEnd);
        }
    }

    function update(next, blocked = setupBusy) {
        setupBusy = blocked;
        const previous = job()?.phase;
        if (next !== undefined) doc = next;
        const current = job(), phase = current?.phase;
        if (phase === 'ready' && previous !== 'ready') {
            if (drafts.get('shared/text_encoder') === (doc.shared_encoder || 'builtin')) drafts.delete('shared/text_encoder');
            if (previous) changed();
        }
        jobRow.hidden = !phase;
        cancel.hidden = !verifying();
        verification.textContent = ({
            waiting_for_gpu: t('Waiting for the GPU to verify the encoder…', '正在等待 GPU 验证编码器…'),
            smoke: t('Verifying the encoder…', '正在验证编码器…'),
            cancelling: t('Cancelling verification…', '正在取消验证…'),
            ready: t('Shared encoder saved.', '共享编码器已保存。'),
            cancelled: t('Verification cancelled.', '验证已取消。'),
            failed: current?.error || t('Encoder verification failed. Save again to retry.', '编码器验证失败，请再次保存以重试。'),
        })[phase] || '';
        paint();
    }

    cancel.onclick = () => run(async () => {
        try { await request('image-model-cancel', {}); void refresh?.(); }
        catch (error) { say(error?.message || t('Could not cancel verification.', '未能取消验证。'), 'warn'); }
    });
    return {element, update, busy: () => working || verifying()};
}
