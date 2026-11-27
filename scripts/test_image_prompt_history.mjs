import assert from 'node:assert/strict';
import test from 'node:test';
import {createImageGenClient} from '../web/image_generation.js';
import {forgetDeletedPrompts, HISTORY_KEY, PRESET_KEY, rememberPrompt} from '../web/image_view_prompts.js';

test('image client sends the display prompt and busy retry opt-in independently of the generation prompt', async () => {
    const sent = [];
    const client = createImageGenClient(async (path, options) => {
        if (path === '/freevideo/image-gen') return {ok: true, json: async () => ({token: 'token'})};
        assert.equal(path, '/freevideo/image-gen/generate');
        sent.push(JSON.parse(options.body));
        return {ok: true, json: async () => ({job: 'job'})};
    });
    const input = {prompt: 'A cat in <Picture 1>', width: 256, height: 256};
    await client.generate({...input, history_prompt: 'A cat in @Image1', retry_busy: true});
    await client.generate(input);

    assert.deepEqual(sent, [
        {...input, images: [], history_prompt: 'A cat in @Image1', retry_busy: true},
        {...input, images: []},
    ]);
});

test('forget removes only the versions sent for cleanup while keeping presets and prompts used again', async t => {
    const storage = new Map();
    const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    Object.defineProperty(globalThis, 'localStorage', {configurable: true, value: {
        getItem: key => storage.get(key),
        setItem: (key, value) => storage.set(key, value),
    }});
    t.after(() => {
        if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
        else delete globalThis.localStorage;
    });

    const old = ['removed', 'used again', 'preset'].map(text => ({text, at: 250, count: 1}));
    storage.set(HISTORY_KEY, JSON.stringify(old));
    storage.set(PRESET_KEY, JSON.stringify([{prompt: 'preset'}]));
    const removed = await forgetDeletedPrompts({
        offered: async () => true,
        fetchImpl: async (path, options) => {
            assert.equal(path, '/freevideo/private/forgotten');
            assert.deepEqual(JSON.parse(options.body), {prompts: old.map(row => row.text)});
            storage.set(HISTORY_KEY, JSON.stringify(rememberPrompt(old, 'used again', {at: 250})));
            return {ok: true, json: async () => ({forget: [0, 1, 2]})};
        },
    });

    assert.equal(removed, 1);
    assert.deepEqual(JSON.parse(storage.get(HISTORY_KEY)).map(row => row.text), ['used again', 'preset']);
});
