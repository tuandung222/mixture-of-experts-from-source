// Validate all Mermaid blocks in docs/ by parsing with the mermaid lib.
// Exits non-zero on first parse error so npm run verify can include it later.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
// Set up minimal browser-like globals so mermaid + dompurify load in Node.
import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.Element = dom.window.Element;
globalThis.DocumentFragment = dom.window.DocumentFragment;

const { default: mermaid } = await import('mermaid');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const docsRoot = path.resolve(__dirname, '..', 'docs');

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(p));
    else if (entry.isFile() && p.endsWith('.md')) out.push(p);
  }
  return out;
}

function extractBlocks(content) {
  const blocks = [];
  const re = /```mermaid\n([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(content)) !== null) {
    const before = content.slice(0, m.index);
    const line = before.split('\n').length;
    blocks.push({ src: m[1], startLine: line });
  }
  return blocks;
}

mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });

const files = walk(docsRoot);
let total = 0;
let failed = 0;
for (const file of files) {
  const content = fs.readFileSync(file, 'utf8');
  const blocks = extractBlocks(content);
  for (const { src, startLine } of blocks) {
    total++;
    try {
      await mermaid.parse(src);
    } catch (err) {
      failed++;
      const rel = path.relative(path.resolve(__dirname, '..'), file);
      console.error(`\n[FAIL] ${rel}:${startLine}`);
      console.error(err.message || err);
    }
  }
}

console.log(`\nMermaid blocks checked: ${total}, failed: ${failed}`);
process.exit(failed === 0 ? 0 : 1);
