import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { lookupUniProt } from '../web/core.js';

const html = await readFile(new URL('../web/index.html', import.meta.url), 'utf8');
const app = await readFile(new URL('../web/app.js', import.meta.url), 'utf8');

test('required literal DOM bindings exist while descriptive copy can be omitted', () => {
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  for (const [, id, optional] of app.matchAll(/\$\('([^']+)'\)(\?\.)?/g)) {
    if (!optional) assert.ok(ids.has(id), `Required page element #${id} is missing.`);
  }
});

test('UniProt UI shows the actual rejection reason and preserves the existing sequence', async () => {
  const handlerCode = app.match(/\$\('uniprot-lookup'\)\.addEventListener\('click', async \(\) => \{[\s\S]*?\n\}\);/)?.[0];
  assert.ok(handlerCode, 'UniProt lookup handler is missing.');
  for (const [response, expected] of [
    [new Response('>sp|Q8WZ42|TITIN_HUMAN\n' + 'M'.repeat(34350)), /UniProt entry found.*34,350 residues.*3–1,022 residues/],
    [new Response('', { status: 404 }), /No UniProt entry found/],
  ]) {
    let handler;
    let reportedError;
    const nodes = {
      'uniprot-lookup': { addEventListener: (_, listener) => { handler = listener; } },
      'uniprot-id': { value: 'TITIN_HUMAN' },
      'uniprot-status': { textContent: '' },
      error: { hidden: true },
      sequence: { value: 'MEEPQSDPSV' },
    };
    const sandbox = {
      $: id => nodes[id],
      lookupBusy: false,
      lookupUniProt: identifier => lookupUniProt(identifier, async () => response),
      updateControls() {},
      updateCount() { assert.fail('Rejected lookup must not update the sequence.'); },
      showError(error) { reportedError = error; },
    };
    vm.runInNewContext(handlerCode, sandbox);
    await handler();
    assert.match(nodes['uniprot-status'].textContent, expected);
    assert.match(nodes['uniprot-status'].textContent, /The existing sequence was not changed\.$/);
    assert.equal(nodes.sequence.value, 'MEEPQSDPSV');
    assert.ok(reportedError);
    assert.equal(sandbox.lookupBusy, false);
  }
});

test('analytics initializes the requested ID without custom input events and disclosure stays in the footer', () => {
  const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].find(match => match[1].includes("gtag('config'"))?.[1];
  assert.ok(code, 'Analytics initialization is missing.');
  const sandbox = {};
  sandbox.window = sandbox;
  vm.runInNewContext(code, sandbox);
  assert.equal(sandbox.dataLayer.length, 2);
  assert.equal(sandbox.dataLayer[0][0], 'js');
  assert.deepEqual(Array.from(sandbox.dataLayer[1]), ['config', 'G-DDN4WNM253']);
  assert.match(html, /<script async src="https:\/\/www\.googletagmanager\.com\/gtag\/js\?id=G-DDN4WNM253"><\/script>/);
  assert.match(html, /<footer>[\s\S]*?Google Analytics[\s\S]*?<\/footer>/);
  assert.equal((html.match(/to measure visits and usage/g) || []).length, 1);
});
