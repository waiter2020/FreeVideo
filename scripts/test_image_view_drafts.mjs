import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {createPromptEnhancer, PromptVersions} from '../web/prompt_enhance.js';
import {mentionsToEngine, posePrompt, renumberMentions, sketchPrompt, SUBJECT_WORDS} from '../web/image_view_mentions.js';

const source = readFileSync(new URL('../web/image_view.js', import.meta.url), 'utf8');
const engineText = source.slice(source.indexOf('    function engineText('), source.indexOf('    // What the rewrite sees:'));
const generate = source.slice(source.indexOf('    async function generate()'), source.indexOf('    // Shows a job of the store'));
// Execute the page's actual assembly and submission functions with controllable async boundaries.
const run = new Function('page', `with (page) { ${engineText} ${generate} return generate(); }`);

function pageWith(text) {
    const page = {
        calls: [], notes: [], discarded: [], references: [],
        model: {id: 'qwen-image-2.1-turbo'}, tier: 'standard', width: 1024, height: 1024,
        alive: true, running: null, layoutOn: false, seedInput: {value: '', checkValidity: () => true},
        prompt: {value: text, focus() {}}, note: {dataset: {}}, level: {dataset: {}, style: {setProperty() {}}},
        progressLine: {}, progressSub: {}, runReport: {dataset: {}, replaceChildren() {}}, doneReport: {replaceChildren() {}},
        failure: null, lastReport: null, IMAGE_BUSY: new Set(),
        over: () => false, cap_: () => 4, layoutActive: () => false,
        loraRow: {value: () => []},
        enhancer: {draftRevision: 0, async beforeGenerate() {}, fallbackToOriginal: () => false},
        size: () => ({width: page.width, height: page.height}),
        enhanceScene: () => ({model: page.model.id, references: page.references.map(row => row.file)}),
        client: {upload: async file => file, async discardUploads(names) { page.discarded.push(...names); }},
        jobs: {async submit(kind, request, {meta}) { page.calls.push({kind, request, meta}); return {id: 'picture-job'}; }},
        t: en => en, say: message => { page.notes.push(message); },
        render() {}, drawWorks() {}, clearStep() {}, startJobSources() {}, follow() {},
        mentionsToEngine, SUBJECT_WORDS, posePrompt, sketchPrompt,
        setTimeout: callback => { queueMicrotask(callback); },
    };
    return page;
}

test('replacement starts a new draft and reference numbers follow both versions', () => {
    const original = 'Use @Image1 and 图 1';
    const enhanced = 'Use <Picture 1>, Picture 1, Image 1 and @图1';
    const versions = new PromptVersions(null, original);
    versions.value.enabled = true;
    const a = {}, b = {};
    const context = {references: [1]};
    assert.equal(versions.accept(versions.key(context), context, enhanced), true);
    versions.mapDrafts(text => renumberMentions(text, [a], [b, a]).text);
    assert.equal(versions.value.original, 'Use @Image2 and 图 2');
    assert.equal(versions.value.rewritten, 'Use <Picture 2>, Picture 2, Image 2 and @图2');
    assert.equal(versions.fresh(context), false);
    assert.equal(versions.value.selected, 'rewritten');
    assert.equal(renumberMentions('@图2', [b, a], [b]).text, '@图?');
    versions.replaceDraft('A new scene from a saved prompt');
    assert.equal(versions.text(), 'A new scene from a saved prompt');
    assert.equal(versions.value.rewritten, '');
    assert.equal(versions.value.enabled, true);
});

test('late rewrite cannot replace a new draft and releases media after the worker finishes', async () => {
    class Element extends EventTarget {
        dataset = {};
        style = {};
        classList = {add() {}};
        append() {}
        setAttribute() {}
        removeAttribute() {}
    }
    globalThis.document = {createElement: () => new Element()};
    const input = new Element();
    input.value = 'Earlier draft';
    const node = {id: 'image-view', properties: {freevideo_prompt_versions: {enabled: true, original: input.value, rewritten: ''}}};
    let statusReached, reply;
    const requestedStatus = new Promise(resolve => { statusReached = resolve; });
    const answer = new Promise(resolve => { reply = resolve; });
    const releases = [];
    let enhancer;
    try {
        enhancer = createPromptEnhancer({node, input, t: en => en, setText() {}, context: () => ({model: 'qwen'}),
            discardStale: true, resolve: async () => ({media: [{file: 'upload'}]}), releaseMedia: body => releases.push(body),
            api: {addEventListener() {}, removeEventListener() {}, async fetchApi(path) {
                let value = {ready: true, token: 'local'};
                if (path.endsWith('/rewrite')) value = {job: 'rewrite-job'};
                if (path.endsWith('/status')) { statusReached(); value = await answer; }
                return {ok: true, json: async () => value};
            }},
        });
        const revision = enhancer.draftRevision;
        const pending = enhancer.beforeGenerate({isCurrent: () => enhancer.draftRevision === revision});
        await requestedStatus;
        enhancer.replaceDraft('Later draft');
        assert.deepEqual(releases, []);
        reply({phase: 'complete', text: 'Earlier rewrite'});
        assert.equal(await pending, false);
        assert.equal(input.value, 'Later draft');
        assert.equal(node.properties.freevideo_prompt_versions.rewritten, '');
        assert.deepEqual(releases, [{media: [{file: 'upload'}]}]);
    } finally {
        enhancer?.dispose();
        delete globalThis.document;
    }
});

test('generation checks the selected wire prompt after enhancement in Unicode code points', async () => {
    const page = pageWith('x'.repeat(2001));
    page.enhancer.beforeGenerate = async () => { page.prompt.value = '🌄'.repeat(2000); };
    await run(page);
    assert.equal(page.calls.length, 1);
    assert.equal([...page.calls[0].request.prompt].length, 2000);

    const excess = pageWith('🌄'.repeat(2001));
    await run(excess);
    assert.equal(excess.calls.length, 0);
    assert.match(excess.notes.at(-1), /over 2,000/);
});

test('a valid original replaces an overlong expanded rewrite', async () => {
    const page = pageWith('Earlier');
    page.references = [{file: 'A'}];
    page.enhancer.beforeGenerate = async () => { page.prompt.value = `@Image1 ${'x'.repeat(1990)}`; };
    page.enhancer.fallbackToOriginal = usable => {
        assert.equal(usable('A short caption with @Image1'), true);
        page.prompt.value = 'A short caption with @Image1';
        return true;
    };
    await run(page);
    assert.equal(page.calls[0].request.prompt, 'A short caption with <Picture 1>');
    assert.equal(page.calls[0].request.history_prompt, 'A short caption with @Image1');
    assert.equal(page.calls[0].meta.text, page.calls[0].request.history_prompt);
});

test('a draft changed during enhancement never starts uploading to generate', async () => {
    const page = pageWith('The original view');
    page.references = [{file: 'A'}];
    page.client.upload = async () => assert.fail('a stale draft cannot upload images for generation');
    page.enhancer.beforeGenerate = async ({isCurrent}) => {
        page.width = 768;
        assert.equal(isCurrent(), false);
    };
    await run(page);
    assert.equal(page.calls.length, 0);
    assert.match(page.notes.at(-1), /settings changed/);
});

test('a generation upload invalidates the request if the user swaps a reference or model', async () => {
    const page = pageWith('An edit based on @Image1');
    page.references = [{file: 'old'}];
    page.client.upload = async file => {
        page.references[0].file = 'new';
        page.model = {id: 'minimax-h3-image'};
        return file;
    };
    await run(page);
    assert.equal(page.calls.length, 0);
    assert.deepEqual(page.discarded, ['old']);
    assert.match(page.notes.at(-1), /settings changed/);
});

test('busy retries reuse the complete request and history after the page changes', async () => {
    const page = pageWith('Use @Image1');
    page.references = [{file: 'first'}];
    const submit = page.jobs.submit;
    page.jobs.submit = async (...args) => {
        const result = await submit(...args);
        if (page.calls.length === 1) {
            page.prompt.value = 'Another subject';
            page.model = {id: 'minimax-h3-image'};
            page.references[0].file = 'second';
            throw Object.assign(new Error('busy'), {code: 'busy'});
        }
        return result;
    };
    await run(page);
    assert.equal(page.calls.length, 2);
    const [first, second] = page.calls;
    assert.equal(first.request, second.request);
    assert.equal(first.meta, second.meta);
    assert.deepEqual(first.request.images, ['first']);
    assert.equal(first.request.prompt, 'Use <Picture 1>');
    assert.equal(first.request.model, 'qwen-image-2.1-turbo');
    assert.equal(first.request.retry_busy, true);
    assert.deepEqual(page.discarded, []);
});

test('ending a busy wait cleans the uploads retained for retry', async () => {
    const page = pageWith('Use @Image1');
    page.references = [{file: 'saved-for-retry'}];
    page.jobs.submit = async () => {
        page.running.cancelled = true;
        throw Object.assign(new Error('busy'), {code: 'busy'});
    };
    await run(page);
    assert.deepEqual(page.discarded, ['saved-for-retry']);
});
