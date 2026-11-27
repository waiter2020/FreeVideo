import assert from 'node:assert/strict';
import test from 'node:test';
import {createPiVideoPacks, piVideoPackState} from '../web/pi_video_packs.js';

const response = (value, ok = true) => ({ok, json: async () => structuredClone(value)});
const en = (text, _, values = {}) => text.replace(/\{(\w+)\}/g, (original, key) => values[key] ?? original);
const missing = {present: false, published: true, compatible: true, bytes: 100, install: {state: 'idle'}};

test('Settings and generation share download state, job-targeted pause, and pack readiness', async () => {
    const server = {fast2: missing, fast4: missing};
    const commands = [];
    let jobNumber = 0;
    const packs = createPiVideoPacks(async (path, options) => {
        if (options.method !== 'POST') return response(server[path.split('=')[1]]);
        const body = JSON.parse(options.body);
        commands.push(body);
        server[body.mode] = body.action === 'install'
            ? {...server[body.mode], install: {state: 'running', done: 20, total: 100, job_id: `job-${++jobNumber}`}}
            : {...server[body.mode], install: {...server[body.mode].install, cancelling: true}};
        return response(server[body.mode]);
    });
    await packs.refresh();
    const states = {settings: [], picker: []};
    const observe = view => packs.subscribe(() => states[view].push(piVideoPackState(en, packs.status('fast2')).state));
    const closeSettings = observe('settings'), closePicker = observe('picker');
    try {
        await packs.action('fast2', 'install');
        const firstRequest = packs.ensure('fast2');
        assert.equal(piVideoPackState(en, packs.status('fast4')).state, 'blocked');
        server.fast2 = {...server.fast2, install: {...server.fast2.install, phase: 'verify', verify_done: 30, verify_total: 60}};
        await packs.refresh();
        const verifying = piVideoPackState(en, packs.status('fast2'));
        assert.equal(verifying.verifying, true);
        assert.equal(verifying.fraction, 0.5);
        await packs.action('fast2', 'cancel');
        assert.equal(piVideoPackState(en, packs.status('fast2')).cancelling, true);
        assert.deepEqual(commands, [{action: 'install', mode: 'fast2'}, {action: 'cancel', mode: 'fast2', job_id: 'job-1'}]);
        server.fast2 = {...missing, resume_bytes: 20, install: {state: 'cancelled'}};
        await packs.refresh();
        assert.equal(await firstRequest, false);
        assert.equal(piVideoPackState(en, packs.status('fast2')).resume, true);
        assert.equal(piVideoPackState(en, packs.status('fast4')).state, 'missing');
        closeSettings();
        const resumed = packs.ensure('fast2');
        await packs.read('fast2');
        server.fast2 = {...missing, present: true, bytes: 0, install: {state: 'done', job_id: 'job-2'}};
        await packs.refresh();
        assert.equal(await resumed, true);
        assert.ok(states.settings.includes('downloading'));
        assert.equal(states.picker.at(-1), 'ready');
        assert.equal(commands.length, 3);
    } finally { closeSettings(); closePicker(); }
});

test('a poll started before a command cannot overwrite the visible job', async () => {
    let completeRead;
    const packs = createPiVideoPacks((_, options) => {
        if (options.method === 'POST') return response({...missing, install: {state: 'running', job_id: 'new'}});
        return new Promise(resolve => { completeRead = () => resolve(response(missing)); });
    });
    const previous = packs.read('fast4');
    const action = await packs.action('fast4', 'install');
    assert.equal(action.install.job_id, 'new');
    completeRead();
    await previous;
    assert.equal(packs.status('fast4').install.job_id, 'new');
});

test('missing, interrupted, failed, and incompatible packs keep distinct management states', () => {
    assert.deepEqual(piVideoPackState(en, undefined), {state: 'checking'});
    const interrupted = piVideoPackState(en, {...missing, resume_bytes: 40, install: {state: 'failed', reason: 'interrupted'}});
    assert.equal(interrupted.state, 'missing');
    assert.equal(interrupted.resume, true);
    assert.equal(interrupted.resumeBytes, 40);
    assert.match(interrupted.reason, /interrupted/);
    const failed = piVideoPackState(en, {...missing, install: {state: 'failed', reason: 'integrity_failed'}});
    assert.equal(failed.state, 'failed');
    assert.match(failed.reason, /verification/);
    const incompatible = piVideoPackState(en, {...missing, present: true, compatible: false, compatibility_reason: 'gpu'});
    assert.equal(incompatible.state, 'unavailable');
    assert.equal(incompatible.present, true);
    assert.match(incompatible.reason, /NVIDIA/);
});
