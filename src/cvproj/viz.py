"""Lightweight self-contained 2D/3D Three.js visualization of input, reconstruction, and error."""

import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvproj.model import Autoencoder

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MRI Reconstruction & Error Viewer — __SUBJECT__</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: system-ui, sans-serif; padding: 14px; }
  header {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
    justify-content: space-between; background: #1a1d27; padding: 10px 16px;
    border-radius: 8px; margin-bottom: 12px; border: 1px solid #2d3348;
  }
  .title { font-weight: 600; font-size: 15px; color: #f8fafc; }
  .meta { font-size: 13px; color: #94a3b8; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }
  .controls label { display: flex; align-items: center; gap: 6px; color: #cbd5e1; }
  select, input[type=range] { accent-color: #38bdf8; cursor: pointer; }
  select {
    background: #0f1117; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 4px; padding: 3px 6px;
  }
  .grid-2d { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }
  .card {
    background: #1a1d27; border: 1px solid #2d3348;
    border-radius: 8px; padding: 8px; text-align: center;
  }
  .card-title { font-size: 12px; color: #94a3b8; margin-bottom: 6px; font-weight: 500; }
  canvas.slice {
    width: 100%; max-width: 240px; aspect-ratio: 1;
    image-rendering: pixelated; border-radius: 4px; background: #000;
  }
  .grid-3d { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; height: 380px; }
  .viewport-3d {
    position: relative; width: 100%; height: 100%;
    background: #090b10; border-radius: 6px; overflow: hidden;
  }
  .badge { font-family: monospace; color: #38bdf8; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/examples/jsm/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<header>
  <div>
    <div class="title">MRI Reconstruction & Anomaly Residual Inspector</div>
    <div class="meta">
      Subject: <span class="badge" id="meta-subj"></span> |
      Subject MSE: <span class="badge" id="meta-mse"></span> |
      Max |Error|: <span class="badge" id="meta-max"></span> |
      Grid: <span class="badge" id="meta-shape"></span>
    </div>
  </div>
  <div class="controls">
    <label id="chan-wrap">Channel: <select id="chan-select"></select></label>
    <label>Slice (<span id="slice-val">0</span>):
      <input type="range" id="slice-slider" min="0" max="1" value="0">
    </label>
    <label>Error Highlight Thresh (<span id="thresh-val">0.15</span>):
      <input type="range" id="thresh-slider" min="0.02" max="0.80" step="0.02" value="0.15">
    </label>
    <label>3D Mode:
      <select id="style-select">
        <option value="mip">MIP (Max Intensity)</option>
        <option value="iso">ISO Surface</option>
      </select>
    </label>
  </div>
</header>

<div class="grid-2d">
  <div class="card">
    <div class="card-title">Input Slice (X)</div>
    <canvas id="c-inp" class="slice"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Reconstruction (X̂)</div>
    <canvas id="c-rec" class="slice"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Residual Error |X - X̂| (Scaled)</div>
    <canvas id="c-err" class="slice"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Input + Error Highlight Overlay</div>
    <canvas id="c-ovl" class="slice"></canvas>
  </div>
</div>

<div class="grid-3d">
  <div class="card" style="display:flex;flex-direction:column;">
    <div class="card-title">
      3D Volume Raycast:
      <select id="vol-mode">
        <option value="inp">Input X</option>
        <option value="rec">Reconstruction X̂</option>
      </select>
    </div>
    <div id="v3d-brain" class="viewport-3d"></div>
  </div>
  <div class="card" style="display:flex;flex-direction:column;">
    <div class="card-title">3D Reconstruction Error Cloud |X - X̂| (Synced OrbitControls)</div>
    <div id="v3d-error" class="viewport-3d"></div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VolumeRenderShader1 } from 'three/examples/jsm/shaders/VolumeShader.js';

const PAYLOAD = __PAYLOAD_JSON__;

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255.0;
  return out;
}

const { subject, mse, max_error, channels, shape, inp_b64, rec_b64, err_b64 } = PAYLOAD;
const [C, D, H, W] = shape;
const inp = b64ToFloat32(inp_b64);
const rec = b64ToFloat32(rec_b64);
const err = b64ToFloat32(err_b64);

document.getElementById('meta-subj').textContent = subject;
document.getElementById('meta-mse').textContent = mse.toFixed(6);
document.getElementById('meta-max').textContent = max_error.toFixed(4);
document.getElementById('meta-shape').textContent = `${W}x${H}x${D} (${C}ch)`;

const chanSelect = document.getElementById('chan-select');
channels.forEach((name, idx) => {
  const opt = document.createElement('option');
  opt.value = idx;
  opt.textContent = name;
  chanSelect.appendChild(opt);
});
if (C === 1) document.getElementById('chan-wrap').style.display = 'none';

const sliceSlider = document.getElementById('slice-slider');
sliceSlider.max = Math.max(0, D - 1);
sliceSlider.value = Math.floor(D / 2);

const threshSlider = document.getElementById('thresh-slider');
const styleSelect = document.getElementById('style-select');
const volModeSelect = document.getElementById('vol-mode');

function hotColor(t) {
  const r = Math.min(1, Math.max(0, t * 2.5));
  const g = Math.min(1, Math.max(0, (t - 0.35) * 2.2));
  const b = Math.min(1, Math.max(0, (t - 0.75) * 4.0));
  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
}

function render2D() {
  const c = parseInt(chanSelect.value, 10) || 0;
  const z = parseInt(sliceSlider.value, 10) || 0;
  const thresh = parseFloat(threshSlider.value);
  document.getElementById('slice-val').textContent = z;
  document.getElementById('thresh-val').textContent = thresh.toFixed(2);

  const ids = ['c-inp', 'c-rec', 'c-err', 'c-ovl'];
  const ctxs = ids.map(id => {
    const cv = document.getElementById(id);
    cv.width = W; cv.height = H;
    return cv.getContext('2d');
  });
  const imgs = ctxs.map(ctx => ctx.createImageData(W, H));
  const offset = (c * D + z) * H * W;

  for (let i = 0; i < H * W; i++) {
    const vInp = inp[offset + i];
    const vRec = rec[offset + i];
    const vErr = err[offset + i];
    const gInp = Math.floor(vInp * 255);
    const gRec = Math.floor(vRec * 255);
    const [hr, hg, hb] = hotColor(vErr);
    const p = i * 4;

    imgs[0].data.set([gInp, gInp, gInp, 255], p);
    imgs[1].data.set([gRec, gRec, gRec, 255], p);
    imgs[2].data.set([hr, hg, hb, 255], p);

    if (vErr >= thresh) {
      const alpha = Math.min(0.85, 0.35 + (vErr - thresh) * 1.2);
      imgs[3].data.set([
        Math.floor((1 - alpha) * gInp + alpha * 255),
        Math.floor((1 - alpha) * gInp + alpha * Math.max(0, (vErr - thresh) * 180)),
        Math.floor((1 - alpha) * gInp),
        255
      ], p);
    } else {
      imgs[3].data.set([gInp, gInp, gInp, 255], p);
    }
  }
  ctxs.forEach((ctx, idx) => ctx.putImageData(imgs[idx], 0, 0));
}

function makeColormapTexture(mode) {
  const cv = document.createElement('canvas');
  cv.width = 256; cv.height = 1;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const [r, g, b] = mode === 'hot' ? hotColor(t) : [i, i, i];
    img.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.needsUpdate = true;
  return tex;
}

function extract3DVolume(source, c, zRepeat) {
  const D3 = D * zRepeat;
  const out = new Float32Array(W * H * D3);
  const chOffset = c * D * H * W;
  for (let z = 0; z < D3; z++) {
    const zPos = (z / Math.max(1, D3 - 1)) * Math.max(0, D - 1);
    const z0 = Math.floor(zPos), z1 = Math.min(D - 1, z0 + 1);
    const w1 = zPos - z0, w0 = 1 - w1;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx2d = y * W + x;
        const v0 = source[chOffset + z0 * H * W + idx2d];
        const v1 = source[chOffset + z1 * H * W + idx2d];
        out[z * H * W + (H - 1 - y) * W + x] = w0 * v0 + w1 * v1;
      }
    }
  }
  return { data: out, D3 };
}

function init3DPair() {
  const cBrain = document.getElementById('v3d-brain');
  const cError = document.getElementById('v3d-error');
  const zRepeat = Math.max(1, Math.round(W / Math.max(1, D)));
  const D3 = D * zRepeat;

  function createPane(container, cmMode) {
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const aspect = container.clientWidth / Math.max(1, container.clientHeight);
    const frustum = Math.max(W, H, D3) * 1.2;
    const camera = new THREE.OrthographicCamera(
      -frustum * aspect / 2, frustum * aspect / 2, frustum / 2, -frustum / 2, 1, 2000
    );
    camera.position.set(W * 1.5, -H * 1.8, D3 * 1.4);
    camera.up.set(0, 0, 1);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(W / 2, H / 2, D3 / 2);
    controls.update();

    const dummy = new Float32Array(W * H * D3);
    const tex = new THREE.Data3DTexture(dummy, W, H, D3);
    tex.format = THREE.RedFormat;
    tex.type = THREE.FloatType;
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.unpackAlignment = 1;

    const uniforms = THREE.UniformsUtils.clone(VolumeRenderShader1.uniforms);
    uniforms['u_data'].value = tex;
    uniforms['u_size'].value.set(W, H, D3);
    uniforms['u_renderstyle'].value = 0;
    uniforms['u_renderthreshold'].value = 0.15;
    uniforms['u_cmdata'].value = makeColormapTexture(cmMode);

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: VolumeRenderShader1.vertexShader,
      fragmentShader: VolumeRenderShader1.fragmentShader,
      side: THREE.BackSide,
      transparent: true
    });
    const geom = new THREE.BoxGeometry(W, H, D3);
    geom.translate(W / 2 - 0.5, H / 2 - 0.5, D3 / 2 - 0.5);
    scene.add(new THREE.Mesh(geom, material));

    return { scene, renderer, camera, controls, tex, uniforms, container };
  }

  const left = createPane(cBrain, 'gray');
  const right = createPane(cError, 'hot');

  let syncing = false;
  function sync(src, dst) {
    src.controls.addEventListener('change', () => {
      if (syncing) return;
      syncing = true;
      dst.camera.position.copy(src.camera.position);
      dst.camera.zoom = src.camera.zoom;
      dst.camera.updateProjectionMatrix();
      dst.controls.target.copy(src.controls.target);
      dst.controls.update();
      syncing = false;
    });
  }
  sync(left, right);
  sync(right, left);

  function updateVolumes() {
    const c = parseInt(chanSelect.value, 10) || 0;
    const srcBrain = volModeSelect.value === 'rec' ? rec : inp;
    left.tex.image.data.set(extract3DVolume(srcBrain, c, zRepeat).data);
    left.tex.needsUpdate = true;
    right.tex.image.data.set(extract3DVolume(err, c, zRepeat).data);
    right.tex.needsUpdate = true;

    const style = styleSelect.value === 'mip' ? 0 : 1;
    const thresh = parseFloat(threshSlider.value);
    left.uniforms['u_renderstyle'].value = style;
    right.uniforms['u_renderstyle'].value = style;
    left.uniforms['u_renderthreshold'].value = thresh;
    right.uniforms['u_renderthreshold'].value = thresh;
  }

  function animate() {
    requestAnimationFrame(animate);
    left.renderer.render(left.scene, left.camera);
    right.renderer.render(right.scene, right.camera);
  }
  updateVolumes();
  animate();
  return updateVolumes;
}

render2D();
const update3D = init3DPair();
[chanSelect, sliceSlider, threshSlider].forEach(el => el.addEventListener('input', render2D));
[chanSelect, threshSlider, styleSelect, volModeSelect].forEach(
  el => el.addEventListener('input', update3D)
);
</script>
</body>
</html>
"""


def _encode_uint8(array: np.ndarray) -> str:
    clipped = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return base64.b64encode(clipped.tobytes()).decode("ascii")


def save_visualization(
    model: Autoencoder,
    data: DataLoader,
    device: torch.device,
    output_path: Path,
    modalities: list[str],
) -> Path:
    """Save a self-contained 2D/3D Three.js HTML visualization for the first subject in `data`."""
    model.eval()
    target_subject: str | None = None
    inp_slices: list[np.ndarray] = []
    rec_slices: list[np.ndarray] = []
    mse_values: list[float] = []

    with torch.inference_mode():
        for images, subjects in data:
            images = images.to(device)
            preds = model(images)
            mask = images > 0
            for idx, subj in enumerate(subjects):
                if target_subject is None:
                    target_subject = str(subj)
                if str(subj) != target_subject:
                    continue
                x = images[idx]
                y = preds[idx]
                m = mask[idx]
                denom = int(m.sum().item())
                if denom > 0:
                    mse_values.append(float(((y - x).square() * m).sum().item() / denom))
                inp_slices.append(x.cpu().numpy())
                rec_slices.append((y * m).cpu().numpy())
            if target_subject is not None and all(str(s) != target_subject for s in subjects):
                break

    if not inp_slices or target_subject is None:
        raise ValueError("No slices available to visualize")

    # Stack to shape (C, D, H, W)
    inp_vol = np.stack(inp_slices, axis=1).astype(np.float32)
    rec_vol = np.stack(rec_slices, axis=1).astype(np.float32)
    err_raw = np.abs(inp_vol - rec_vol) * (inp_vol > 0)
    max_err = float(err_raw.max()) if err_raw.size else 0.0
    err_norm = err_raw / max(max_err, 1e-6)

    payload = {
        "subject": target_subject,
        "mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "max_error": max_err,
        "channels": modalities,
        "shape": list(inp_vol.shape),
        "inp_b64": _encode_uint8(inp_vol),
        "rec_b64": _encode_uint8(rec_vol),
        "err_b64": _encode_uint8(err_norm),
    }
    html = HTML_TEMPLATE.replace("__SUBJECT__", target_subject).replace(
        "__PAYLOAD_JSON__", json.dumps(payload)
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
