#!/usr/bin/env node
'use strict';
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const REPO = 'AnEntrypoint/rs-learn';
const pkg = require('./package.json');
const VERSION = `v${pkg.version}`;

const TARGET_MAP = {
  'linux-x64': 'rs-learn-x86_64-unknown-linux-gnu',
  'linux-arm64': 'rs-learn-aarch64-unknown-linux-gnu',
  'darwin-x64': 'rs-learn-x86_64-apple-darwin',
  'darwin-arm64': 'rs-learn-aarch64-apple-darwin',
  'win32-x64': 'rs-learn-x86_64-pc-windows-msvc.exe',
  'win32-arm64': 'rs-learn-aarch64-pc-windows-msvc.exe',
};

const key = `${process.platform}-${process.arch}`;
const binaryName = TARGET_MAP[key];
if (!binaryName) {
  console.error(`rs-learn: unsupported platform ${key}. Install from source: cargo install rs-learn`);
  process.exit(0);
}

const isWin = process.platform === 'win32';
const binDir = path.join(__dirname, 'bin');
const binPath = path.join(binDir, isWin ? 'rs-learn.exe' : 'rs-learn');
const wrapperPath = path.join(binDir, 'rs-learn');

if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });

const url = `https://github.com/${REPO}/releases/download/${VERSION}/${binaryName}`;
console.log(`rs-learn: downloading ${url}`);

function download(url, dest, redirects = 5) {
  return new Promise((resolve, reject) => {
    if (redirects === 0) return reject(new Error('Too many redirects'));
    https.get(url, { headers: { 'User-Agent': 'rs-learn-postinstall' } }, res => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        return resolve(download(res.headers.location, dest, redirects - 1));
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      const file = fs.createWriteStream(dest);
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
      file.on('error', reject);
    }).on('error', reject);
  });
}

async function main() {
  await download(url, binPath);
  fs.chmodSync(binPath, 0o755);

  if (!isWin) {
    fs.writeFileSync(wrapperPath,
      `#!/bin/sh\nexec "$(dirname "$0")/rs-learn" "$@"\n`);
    fs.chmodSync(wrapperPath, 0o755);
  }

  console.log(`rs-learn: installed to ${binPath}`);
}

main().catch(e => {
  console.error(`rs-learn: install failed: ${e.message}`);
  console.error('Install manually: cargo install rs-learn');
  process.exit(0);
});
