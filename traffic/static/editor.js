// editor.js — ROI / counting-line editor bound to a displayed image or video frame.
//
// createEditor({ canvas, image, onChange }) -> {
//   getConfig()   -> { roi: [[x,y],...], line: [[x,y],[x,y]] }   // normalized 0..1
//   setConfig(cfg) -> void   // applies roi/line, redraws
//   setMode(m)    -> void    // 'roi' | 'line' | 'idle'
//   reset()       -> void    // full-frame polygon + default line
//   setEnabled(b) -> void    // false disables all interaction (frozen job)
//   destroy()     -> void
// }
// onChange(config) fires whenever geometry changes.
// Coordinates are normalized against the *displayed* media rectangle so the
// canvas always matches the media aspect ratio (no letterbox mismatch).

export function createEditor({ canvas, image, onChange }) {
  const ctx = canvas.getContext('2d');
  let roi = [[0, 0], [1, 0], [1, 1], [0, 1]];
  let line = [[0.1, 0.5], [0.9, 0.5]];
  let mode = 'idle';
  let enabled = true;
  let pending = []; // in-progress points for roi/line
  let dpr = window.devicePixelRatio || 1;
  let rect = { w: 0, h: 0 };

  const emit = () => { if (typeof onChange === 'function') onChange(getConfig()); };

  function mediaSize() {
    const w = image.naturalWidth || image.videoWidth || image.width || 0;
    const h = image.naturalHeight || image.videoHeight || image.height || 0;
    return { w, h };
  }

  function layout() {
    const { w, h } = mediaSize();
    if (!w || !h) return;
    const box = canvas.parentElement.getBoundingClientRect();
    const availW = Math.max(1, box.width);
    const availH = Math.max(1, box.height);
    const scale = Math.min(availW / w, availH / h);
    const cssW = Math.round(w * scale);
    const cssH = Math.round(h * scale);
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    rect = { w: cssW, h: cssH };
    draw();
  }

  function toNorm(ev) {
    const r = canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width;
    const y = (ev.clientY - r.top) / r.height;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  function draw() {
    if (!rect.w || !rect.h) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.w, rect.h);

    // ROI polygon
    if (roi.length >= 2) {
      ctx.beginPath();
      roi.forEach(([x, y], i) => {
        const px = x * rect.w, py = y * rect.h;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(17,119,108,0.14)';
      ctx.fill();
      ctx.strokeStyle = '#11776c';
      ctx.lineWidth = 2;
      ctx.stroke();
      roi.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * rect.w, y * rect.h, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#11776c';
        ctx.fill();
      });
    }

    // Counting line with A/B side labels
    if (line.length === 2) {
      const [a, b] = line;
      const ax = a[0] * rect.w, ay = a[1] * rect.h;
      const bx = b[0] * rect.w, by = b[1] * rect.h;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = '#e08a2e';
      ctx.lineWidth = 2;
      ctx.stroke();
      // arrow head at b
      const ang = Math.atan2(by - ay, bx - ax);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx - 10 * Math.cos(ang - 0.4), by - 10 * Math.sin(ang - 0.4));
      ctx.lineTo(bx - 10 * Math.cos(ang + 0.4), by - 10 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fillStyle = '#e08a2e';
      ctx.fill();
      ctx.fillStyle = '#20332f';
      ctx.font = '12px "Microsoft YaHei", sans-serif';
      ctx.fillText('A', ax + 6, ay - 6);
      ctx.fillText('B', bx + 6, by - 6);
    }

    // pending points
    pending.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * rect.w, y * rect.h, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#11776c';
      ctx.fill();
    });
  }

  function onClick(ev) {
    if (!enabled) return;
    const p = toNorm(ev);
    if (mode === 'roi') {
      pending.push(p);
      draw();
    } else if (mode === 'line') {
      pending.push(p);
      if (pending.length === 2) {
        line = [pending[0], pending[1]];
        pending = [];
        mode = 'idle';
        draw();
        emit();
      } else {
        draw();
      }
    }
  }

  function onDblClick(ev) {
    if (!enabled || mode !== 'roi') return;
    ev.preventDefault();
    // remove duplicate point from the dblclick's second click
    if (pending.length > 1) pending.pop();
    if (pending.length >= 3) {
      roi = pending.slice();
      pending = [];
      mode = 'idle';
      draw();
      emit();
    }
  }

  function onKey(ev) {
    if (ev.key === 'Escape' && pending.length) {
      pending = [];
      mode = 'idle';
      draw();
    }
  }

  canvas.addEventListener('click', onClick);
  canvas.addEventListener('dblclick', onDblClick);
  window.addEventListener('keydown', onKey);

  const ro = new ResizeObserver(layout);
  ro.observe(canvas.parentElement);
  if (image.tagName === 'VIDEO') {
    image.addEventListener('loadedmetadata', layout);
  } else {
    image.addEventListener('load', layout);
  }
  layout();

  function getConfig() {
    return { roi: roi.map(p => [p[0], p[1]]), line: line.map(p => [p[0], p[1]]) };
  }

  function setConfig(cfg) {
    if (!cfg) return;
    if (Array.isArray(cfg.roi) && cfg.roi.length >= 3) roi = cfg.roi.map(p => [p[0], p[1]]);
    if (Array.isArray(cfg.line) && cfg.line.length === 2) line = cfg.line.map(p => [p[0], p[1]]);
    pending = [];
    draw();
  }

  function setMode(m) {
    mode = m === 'roi' || m === 'line' ? m : 'idle';
    pending = [];
    draw();
  }

  function reset() {
    roi = [[0, 0], [1, 0], [1, 1], [0, 1]];
    line = [[0.1, 0.5], [0.9, 0.5]];
    pending = [];
    mode = 'idle';
    draw();
    emit();
  }

  function setEnabled(b) {
    enabled = !!b;
    canvas.style.cursor = enabled ? 'crosshair' : 'default';
  }

  function destroy() {
    canvas.removeEventListener('click', onClick);
    canvas.removeEventListener('dblclick', onDblClick);
    window.removeEventListener('keydown', onKey);
    ro.disconnect();
  }

  setEnabled(true);
  return { getConfig, setConfig, setMode, reset, setEnabled, destroy };
}
