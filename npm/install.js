'use strict';
const https = require('https');
const fs = require('fs');
const path = require('path');

const REPO = 'AnEntrypoint/rs-learn';
const pkg = require('./package.json');

const TARGET_MAP = {
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'linux-arm64': 'aarch64-unknown-linux-gnu',
  'darwin-x64': 'x86_64-apple-darwin',
  'darwin-arm64': 'aarch64-apple-darwin',
  'win32-x64': 'x86_64-pc-windows-msvc',
  'win32-arm64': 'aarch64-pc-windows-msvc',
};

function targetForHost() {
  const key = `${process.platform}-${process.arch}`;
  const t = TARGET_MAP[key];
  if (!t) throw new Error(`unsupported platform ${key}`);
  return t;
}

function binaryPaths() {
  const isWin = process.platform === 'win32';
  const ext = isWin ? '.exe' : '';
  const vendorDir = path.join(__dirname, 'vendor');
  return {
    isWin,
    binDir: vendorDir,
    binPath: path.join(vendorDir, `rs-learn${ext}`),
    validatePath: path.join(vendorDir, `rs-learn-validate${ext}`),
  };
}

function assetUrl(version, target, which, ext) {
  return `https://github.com/${REPO}/releases/download/v${version}/${which}-${target}${ext}`;
}

function download(url, dest, redirects = 6) {
  return new Promise((resolve, reject) => {
    if (redirects === 0) return reject(new Error('too many redirects'));
    const req = https.get(url, { headers: { 'User-Agent': 'rs-learn-installer' } }, res => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return resolve(download(res.headers.location, dest, redirects - 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      const tmp = `${dest}.part`;
      const file = fs.createWriteStream(tmp);
      res.pipe(file);
      file.on('finish', () => file.close(err => {
        if (err) return reject(err);
        try { fs.renameSync(tmp, dest); resolve(); } catch (e) { reject(e); }
      }));
      file.on('error', err => { try { fs.unlinkSync(tmp); } catch {} reject(err); });
    });
    req.on('error', reject);
    req.setTimeout(60_000, () => { req.destroy(new Error('request timeout')); });
  });
}

async function downloadWithRetry(url, dest, attempts = 5) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try { await download(url, dest); return; }
    catch (e) {
      lastErr = e;
      const delay = Math.min(2000 * Math.pow(2, i), 20_000);
      await new Promise(r => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

async function ensureBinary(opts = {}) {
  const { verbose = false } = opts;
  const { isWin, binDir, binPath, validatePath } = binaryPaths();
  if (fs.existsSync(binPath) && fs.statSync(binPath).size > 0) return binPath;

  const target = targetForHost();
  const version = pkg.version;
  const ext = isWin ? '.exe' : '';
  if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });

  const mainUrl = assetUrl(version, target, 'rs-learn', ext);
  const validateUrl = assetUrl(version, target, 'rs-learn-validate', ext);

  if (verbose) console.log(`rs-learn: downloading ${mainUrl}`);
  await downloadWithRetry(mainUrl, binPath);
  fs.chmodSync(binPath, 0o755);

  try {
    if (verbose) console.log(`rs-learn: downloading ${validateUrl}`);
    await downloadWithRetry(validateUrl, validatePath);
    fs.chmodSync(validatePath, 0o755);
  } catch (e) {
    if (verbose) console.error(`rs-learn: validate binary unavailable (${e.message}); continuing`);
  }

  if (verbose) console.log(`rs-learn: installed ${binPath}`);
  return binPath;
}

module.exports = { ensureBinary, binaryPaths, targetForHost };
