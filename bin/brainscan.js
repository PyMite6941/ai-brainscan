#!/usr/bin/env node
// npm entry point. The tool is Python; this file finds Python, builds a venv
// on first run, installs the package into it, then forwards every argument to
// `python -m brainscan`. No dependencies, so `npm i -g ai-brainscan` stays tiny.
//
//   brainscan install [--gpu] [--extras web,tui,whitebox]   (re)build the venv
//   brainscan --python <args>                               run the venv's python
//   brainscan <anything else>                               -> python -m brainscan
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const PKG_ROOT = path.resolve(__dirname, "..");
const VERSION = require(path.join(PKG_ROOT, "package.json")).version;
const IS_WIN = process.platform === "win32";
const IS_MAC = process.platform === "darwin";

// A clone with its own .venv (made by scripts/install.*) wins over ~/.brainscan.
const LOCAL_VENV = path.join(PKG_ROOT, ".venv");
const HOME = process.env.BRAINSCAN_HOME || path.join(os.homedir(), ".brainscan");
const VENV = fs.existsSync(LOCAL_VENV) ? LOCAL_VENV : path.join(HOME, "venv");
const STAMP = path.join(VENV, ".brainscan-installed");

function venvPython() {
  return IS_WIN ? path.join(VENV, "Scripts", "python.exe") : path.join(VENV, "bin", "python");
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.error) throw r.error;
  return r.status === null ? 1 : r.status;
}

function findPython() {
  const candidates = IS_WIN
    ? [["py", ["-3"]], ["python", []], ["python3", []]]
    : [["python3", []], ["python", []]];
  for (const [cmd, pre] of candidates) {
    const r = spawnSync(cmd, [...pre, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"], {
      encoding: "utf8",
    });
    if (r.status === 0) {
      const [maj, min] = r.stdout.trim().split(".").map(Number);
      if (maj === 3 && min >= 10) return [cmd, pre];
    }
  }
  console.error("ai-brainscan needs Python 3.10+ on PATH. Get it from https://www.python.org/downloads/");
  process.exit(1);
}

function install(argv) {
  const gpu = argv.includes("--gpu");
  const i = argv.indexOf("--extras");
  const extras = i >= 0 && argv[i + 1] ? argv[i + 1] : "web,tui";
  const [py, pre] = findPython();
  fs.mkdirSync(path.dirname(VENV), { recursive: true });
  console.log(`[brainscan] creating venv at ${VENV}`);
  if (run(py, [...pre, "-m", "venv", VENV]) !== 0) process.exit(1);
  const pip = [venvPython(), ["-m", "pip", "install", "--upgrade"]];
  run(pip[0], [...pip[1], "pip"]);
  // torch first, from the right index: CPU wheels unless --gpu (macOS has one wheel).
  const torchArgs = ["torch"];
  if (!IS_MAC && !gpu) torchArgs.push("--index-url", "https://download.pytorch.org/whl/cpu");
  console.log(`[brainscan] installing torch (${IS_MAC ? "macOS" : gpu ? "CUDA" : "CPU"})`);
  if (run(pip[0], [...pip[1], ...torchArgs]) !== 0) process.exit(1);
  console.log(`[brainscan] installing ai-brainscan[${extras}]`);
  if (run(pip[0], [...pip[1], `${PKG_ROOT}[${extras}]`]) !== 0) process.exit(1);
  fs.writeFileSync(STAMP, VERSION);
  console.log("[brainscan] installed. next: brainscan setup   (downloads the default model)");
}

function ensureInstalled() {
  const ok = fs.existsSync(venvPython()) &&
    (VENV === LOCAL_VENV || (fs.existsSync(STAMP) && fs.readFileSync(STAMP, "utf8").trim() === VERSION));
  if (!ok) {
    console.log("[brainscan] first run: setting up the Python side (one time, a few minutes)");
    install([]);
  }
}

function main() {
  const argv = process.argv.slice(2);
  if (argv[0] === "install") return install(argv.slice(1));
  ensureInstalled();
  const args = argv[0] === "--python" ? argv.slice(1) : ["-m", "brainscan", ...argv];
  process.exit(run(venvPython(), args, { env: { ...process.env, PYTHONUTF8: "1" } }));
}

main();
