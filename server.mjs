import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css',
  '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.png': 'image/png',
  '.wasm': 'application/wasm', '.svg': 'image/svg+xml',
};
const port = Number(process.env.PORT || 5173);
http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    const runtime = url.pathname.startsWith('/runtime/');
    const zipLibrary = url.pathname === '/vendor/fflate.js';
    const base = path.join(root, runtime ? 'node_modules\\onnxruntime-web\\dist' : zipLibrary ? 'node_modules\\fflate\\esm' : 'web');
    const relative = zipLibrary ? 'browser.js' : decodeURIComponent(runtime ? url.pathname.slice(9) : url.pathname.slice(1));
    const filename = path.resolve(base, relative || 'index.html');
    if (!filename.startsWith(base + path.sep)) {
      res.writeHead(403).end('Forbidden');
      return;
    }
    const info = await stat(filename);
    if (!info.isFile()) {
      res.writeHead(404).end('Not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': types[path.extname(filename)] || 'application/octet-stream',
      'Content-Length': info.size,
      'Cache-Control': url.pathname.startsWith('/assets/models/') ? 'public, max-age=3600' : 'no-cache',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'X-Content-Type-Options': 'nosniff',
    });
    if (req.method === 'HEAD') res.end();
    else createReadStream(filename).on('error', (error) => res.destroy(error)).pipe(res);
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') res.writeHead(404).end('Not found');
    else {
      console.error(error);
      res.writeHead(500).end('Unable to serve this file');
    }
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Vermeer prototype: http://127.0.0.1:${port}`);
});
