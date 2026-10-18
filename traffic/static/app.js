// app.js — ByteTrack-Traffic experiment comparison UI (vanilla ES modules).
// Wires the accepted static HTML to the real API. No simulated outputs.

import { createEditor } from './editor.js';
import { createPlayer } from './player.js';
import { createTruth } from './truth.js';

const API = '/api';

/* ------------------------------------------------------------------ */
/* DOM references                                                      */
/* ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

const el = {
  // tabs
  tabCompare: $('tab-compare'),
  tabReview: $('tab-review'),

  // upload
  dropTarget: $('drop-target'),
  fileInput: $('file-input'),
  fpsOverride: $('fps-override'),
  videoSelect: $('video-select'),
  videoMeta: $('video-meta'),
  metaFilename: $('meta-filename'),
  metaSize: $('meta-size'),
  metaFps: $('meta-fps'),
  metaFrames: $('meta-frames'),
  metaDuration: $('meta-duration'),

  // geometry
  btnDrawRoi: $('btn-draw-roi'),
  btnDrawLine: $('btn-draw-line'),
  btnResetGeo: $('btn-reset-geo'),
  geoStatus: $('geo-status'),

  // run
  btnRun: $('btn-run'),
  btnCancel: $('btn-cancel'),
  progressWrap: $('progress-wrap'),
  progressBar: $('progress-bar'),
  runStatus: $('run-status'),
  jobSelect: $('job-select'),

  // main
  modeSelect: $('mode-select'),
  compareView: $('compare-view'),
  cardCTitle: $('card-c-title'),

  // stages
  stageA: $('stage-a'),
  frameA: $('frame-a'),
  canvasA: $('canvas-a'),
  videoA: $('video-a'),
  blankA: $('blank-a'),
  stageC: $('stage-c'),
  videoC: $('video-c'),
  blankC: $('blank-c'),

  // transport
  transport: $('transport'),
  btnPlay: $('btn-play'),
  btnPrev: $('btn-prev'),
  btnNext: $('btn-next'),
  timeline: $('timeline'),
  frameReadout: $('frame-readout'),
  speedSelect: $('speed-select'),
  syncIndicator: $('sync-indicator'),

  // table
  countsBody: $('counts-body'),

  // exports
  exportCsv: $('export-csv'),
  exportReport: $('export-report'),
  exportA: $('export-a'),
  exportC: $('export-c'),
  exportDetections: $('export-detections'),

  // review
  reviewView: $('review-view'),

  // new experiment
  btnNewExperiment: $('btn-new-experiment'),

  // toast
  toast: $('toast'),
};

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function setText(node, value) {
  if (node) node.textContent = value == null ? '' : String(value);
}

function show(node, visible) {
  if (!node) return;
  node.hidden = !visible;
}

let toastTimer = null;
function toast(message, kind) {
  if (!el.toast) return;
  el.toast.textContent = message;
  el.toast.classList.toggle('toast-error', kind === 'error');
  el.toast.classList.toggle('toast-ok', kind === 'ok');
  el.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.hidden = true;
  }, 4200);
}

async function apiGet(path) {
  const res = await fetch(API + path, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body && body.error ? String(body.error) : '';
    } catch (_) {
      detail = '';
    }
    throw new Error(detail || `请求失败（${res.status}）`);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const data = await res.json();
      detail = data && data.error ? String(data.error) : '';
    } catch (_) {
      detail = '';
    }
    throw new Error(detail || `请求失败（${res.status}）`);
  }
  return res.json();
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatNumber(value, digits) {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(digits == null ? 2 : digits);
}

/* ------------------------------------------------------------------ */
/* Application state                                                   */
/* ------------------------------------------------------------------ */

const state = {
  health: null,
  videos: [],
  jobs: [],
  classes: [],
  selectedVideo: null,
  selectedJobId: null,
  job: null,
  evaluation: null,
  pollTimer: null,
  uploading: false,
  running: false,
  editor: null,
  player: null,
  truth: null,
  activeTab: 'compare',
  rightMode: 'C',
  frozen: false,
  loadedJobId: null,
  truthDirty: false,
};

/* ------------------------------------------------------------------ */
/* Health / classes                                                    */
/* ------------------------------------------------------------------ */

async function loadHealth() {
  try {
    const health = await apiGet('/health');
    state.health = health;
    state.classes = Array.isArray(health.classes) ? health.classes : [];
    renderClassChips();
    if (health.gpu_available === false) {
      toast('未检测到可用 GPU，运行速度可能较慢', 'error');
    }
  } catch (err) {
    toast(`无法连接服务：${err.message}`, 'error');
  }
}

function renderClassChips() {
  const list = $('class-chips');
  if (!list) return;
  list.textContent = '';
  const classes = state.classes.length
    ? state.classes
    : [
        { id: 0, name: 'car' },
        { id: 1, name: 'bus' },
        { id: 2, name: 'truck' },
        { id: 3, name: 'motorcycle' },
        { id: 4, name: 'bicycle' },
      ];
  for (const cls of classes) {
    const li = document.createElement('li');
    li.className = 'chip';
    li.textContent = cls.name;
    list.appendChild(li);
  }
}

/* ------------------------------------------------------------------ */
/* Videos                                                              */
/* ------------------------------------------------------------------ */

async function loadVideos() {
  try {
    const videos = await apiGet('/videos');
    state.videos = Array.isArray(videos) ? videos : [];
    renderVideoSelect();
  } catch (err) {
    toast(`加载视频列表失败：${err.message}`, 'error');
    renderVideoSelect();
  }
}

function renderVideoSelect() {
  const select = el.videoSelect;
  if (!select) return;
  select.textContent = '';
  if (!state.videos.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '暂无视频';
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '请选择视频';
  select.appendChild(placeholder);
  for (const video of state.videos) {
    const opt = document.createElement('option');
    opt.value = video.video_id;
    opt.textContent = video.filename;
    select.appendChild(opt);
  }
  if (state.selectedVideo) select.value = state.selectedVideo.video_id;
}

function selectVideo(videoId) {
  const video = state.videos.find((v) => v.video_id === videoId) || null;
  state.selectedVideo = video;
  renderVideoMeta();
  updateRunButton();
  if (video) {
    showFirstFrame(video);
  } else {
    clearStages();
  }
}

function renderVideoMeta() {
  const video = state.selectedVideo;
  if (!video) {
    show(el.videoMeta, false);
    return;
  }
  show(el.videoMeta, true);
  setText(el.metaFilename, video.filename);
  setText(el.metaSize, `${video.width} × ${video.height}`);
  setText(el.metaFps, formatNumber(video.fps, 2));
  setText(el.metaFrames, video.frame_count);
  setText(el.metaDuration, formatDuration(video.duration));
}

function showFirstFrame(video) {
  if (!video || !video.frame_url) return;
  el.frameA.src = video.frame_url;
  el.frameA.hidden = false;
  show(el.blankA, false);
  show(el.videoA, false);
  show(el.videoC, false);
  show(el.blankC, true);
  el.canvasA.hidden = false;
  ensureEditor();
}

function clearStages() {
  el.frameA.hidden = true;
  el.frameA.removeAttribute('src');
  el.canvasA.hidden = true;
  show(el.blankA, true);
  show(el.blankC, true);
  show(el.videoA, false);
  show(el.videoC, false);
}

/* ------------------------------------------------------------------ */
/* Upload                                                              */
/* ------------------------------------------------------------------ */

async function handleFile(file) {
  if (!file) return;
  if (state.uploading) return;
  if (state.running) {
    toast('任务运行中，无法上传新视频', 'error');
    return;
  }
  const maxBytes = 512 * 1024 * 1024;
  if (file.size > maxBytes) {
    toast('文件超过 512MB 限制', 'error');
    return;
  }
  state.uploading = true;
  updateRunButton();
  setText(el.runStatus, '正在上传视频…');
  try {
    const form = new FormData();
    form.append('file', file);
    const fpsRaw = el.fpsOverride.value.trim();
    if (fpsRaw) form.append('fps_override', fpsRaw);
    const res = await fetch(`${API}/videos`, { method: 'POST', body: form });
    if (!res.ok) {
      let detail = '';
      try {
        const data = await res.json();
        detail = data && data.error ? String(data.error) : '';
      } catch (_) {
        detail = '';
      }
      throw new Error(detail || `上传失败（${res.status}）`);
    }
    const meta = await res.json();
    toast('视频上传完成', 'ok');
    await loadVideos();
    if (meta && meta.video_id) {
      el.videoSelect.value = meta.video_id;
      selectVideo(meta.video_id);
    }
    setText(el.runStatus, '视频已就绪，可开始对比');
  } catch (err) {
    toast(`上传失败：${err.message}`, 'error');
    setText(el.runStatus, '上传失败');
  } finally {
    state.uploading = false;
    el.fileInput.value = '';
    updateRunButton();
  }
}

/* ------------------------------------------------------------------ */
/* Editor                                                              */
/* ------------------------------------------------------------------ */

function ensureEditor() {
  if (state.editor) return state.editor;
  state.editor = createEditor({
    canvas: el.canvasA,
    image: el.frameA,
    onChange: () => {
      updateGeoStatus();
    },
  });
  updateGeoStatus();
  return state.editor;
}

function updateGeoStatus() {
  if (!state.editor) {
    setText(el.geoStatus, 'ROI：全画面 · 计数线：默认');
    return;
  }
  const cfg = state.editor.getConfig();
  const roiCount = Array.isArray(cfg.roi) ? cfg.roi.length : 0;
  const lineReady = Array.isArray(cfg.line) && cfg.line.length === 2;
  setText(
    el.geoStatus,
    `ROI：${roiCount} 个顶点 · 计数线：${lineReady ? '已设置' : '默认'}`
  );
}

function setEditorEnabled(enabled) {
  if (state.editor) state.editor.setEnabled(enabled);
  el.btnDrawRoi.disabled = !enabled;
  el.btnDrawLine.disabled = !enabled;
  el.btnResetGeo.disabled = !enabled;
}

function guardFrozenGeometry() {
  if (state.frozen) {
    toast('结果已冻结，请点击“新实验”后再修改几何配置', 'error');
    return true;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Jobs                                                                */
/* ------------------------------------------------------------------ */

async function loadJobs() {
  try {
    const jobs = await apiGet('/jobs');
    state.jobs = Array.isArray(jobs) ? jobs : [];
    renderJobSelect();
  } catch (err) {
    toast(`加载任务列表失败：${err.message}`, 'error');
  }
}

function renderJobSelect() {
  const select = el.jobSelect;
  if (!select) return;
  select.textContent = '';
  if (!state.jobs.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '暂无任务';
    select.appendChild(opt);
    return;
  }
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '选择历史任务';
  select.appendChild(placeholder);
  for (const job of state.jobs) {
    const opt = document.createElement('option');
    opt.value = job.job_id;
    const label = job.state ? `${job.job_id} · ${job.state}` : job.job_id;
    opt.textContent = label;
    select.appendChild(opt);
  }
  if (state.selectedJobId) select.value = state.selectedJobId;
}

function updateRunButton() {
  const ready = Boolean(state.selectedVideo) && !state.uploading && !state.running && !state.frozen;
  el.btnRun.disabled = !ready;
}

function updateReviewAvailability() {
  const completed = Boolean(state.job && state.job.state === 'completed');
  if (el.tabReview) {
    el.tabReview.disabled = !completed;
    el.tabReview.setAttribute('aria-disabled', String(!completed));
    el.tabReview.title = completed ? '' : '任务完成后可复核';
  }
  if (!completed && state.activeTab === 'review') {
    switchTab('compare');
  }
}

async function startRun() {
  if (!state.selectedVideo || state.running) return;
  if (state.frozen) {
    toast('当前结果已冻结，请点击“新实验”后再运行', 'error');
    return;
  }
  const editor = ensureEditor();
  const cfg = editor ? editor.getConfig() : null;
  const fpsRaw = el.fpsOverride.value.trim();
  const config = {
    roi: cfg && cfg.roi ? cfg.roi : [],
    line: cfg && cfg.line ? cfg.line : [],
    deadband: 0.005,
    min_confirmations: 2,
    track_thresh: 0.5,
    track_buffer: 30,
    match_thresh: 0.8,
    fps_override: fpsRaw ? Number(fpsRaw) : null,
  };
  state.running = true;
  updateRunButton();
  show(el.btnCancel, true);
  show(el.progressWrap, true);
  setProgress(0);
  setText(el.runStatus, '正在提交任务…');
  try {
    const res = await apiPost('/jobs', {
      video_id: state.selectedVideo.video_id,
      config,
    });
    state.selectedJobId = res.job_id;
    setText(el.runStatus, '任务已提交，正在运行…');
    await loadJobs();
    startPolling(res.job_id);
  } catch (err) {
    state.running = false;
    updateRunButton();
    show(el.btnCancel, false);
    setText(el.runStatus, `提交失败：${err.message}`);
    toast(`提交失败：${err.message}`, 'error');
  }
}

function setProgress(value) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  el.progressBar.style.width = `${pct}%`;
  el.progressBar.setAttribute('aria-valuenow', String(Math.round(pct)));
}

function startPolling(jobId) {
  stopPolling();
  state.pollTimer = setInterval(() => pollJob(jobId), 1000);
  pollJob(jobId);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollJob(jobId) {
  try {
    const job = await apiGet(`/jobs/${jobId}`);
    state.job = job;
    applyJobState(job);
    if (job.state === 'completed' || job.state === 'failed' || job.state === 'cancelled') {
      stopPolling();
      state.running = false;
      updateRunButton();
      show(el.btnCancel, false);
      if (job.state === 'completed') {
        await onJobCompleted(job);
      } else if (job.state === 'failed') {
        setText(el.runStatus, `运行失败：${job.error || '未知错误'}`);
        toast(`运行失败：${job.error || '未知错误'}`, 'error');
      } else {
        setText(el.runStatus, '任务已取消');
      }
    }
  } catch (err) {
    stopPolling();
    state.running = false;
    updateRunButton();
    show(el.btnCancel, false);
    setText(el.runStatus, `状态查询失败：${err.message}`);
  }
}

function applyJobState(job) {
  if (typeof job.progress === 'number') setProgress(job.progress);
  const stage = job.stage ? ` · ${job.stage}` : '';
  if (job.state === 'running' || job.state === 'queued') {
    setText(el.runStatus, `运行中${stage}`);
  }
}

async function onJobCompleted(job) {
  setText(el.runStatus, '对比完成');
  setProgress(100);
  state.frozen = true;
  setEditorEnabled(false);
  await loadEvaluation(job.job_id);
  renderCounts();
  enableExports(true);
  setupPlayerForJob(job);
  updateReviewAvailability();
  toast('对比完成', 'ok');
}

async function cancelRun() {
  if (!state.selectedJobId) return;
  try {
    await apiPost(`/jobs/${state.selectedJobId}/cancel`, {});
    setText(el.runStatus, '正在取消…');
  } catch (err) {
    toast(`取消失败：${err.message}`, 'error');
  }
}

/* ------------------------------------------------------------------ */
/* Evaluation / counts                                                 */
/* ------------------------------------------------------------------ */

async function loadEvaluation(jobId) {
  try {
    state.evaluation = await apiGet(`/jobs/${jobId}/evaluation`);
  } catch (err) {
    state.evaluation = null;
  }
}

function renderCounts() {
  const body = el.countsBody;
  if (!body) return;
  body.textContent = '';
  const evaluation = state.evaluation;
  const reviewed = evaluation && evaluation.status === 'reviewed';
  const reviewer = evaluation && evaluation.reviewer ? evaluation.reviewer : '';
  const modes = evaluation && evaluation.modes ? evaluation.modes : null;
  const aCounts = modes && modes.A && Array.isArray(modes.A.counts) ? modes.A.counts : [];
  const cCounts = modes && modes.C && Array.isArray(modes.C.counts) ? modes.C.counts : [];

  if (!aCounts.length && !cCounts.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6;
    td.className = 'empty-row';
    td.textContent = '暂无计数结果';
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }

  const keys = new Map();
  const collect = (rows) => {
    for (const row of rows) {
      const key = `${row.class_id}|${row.direction}`;
      if (!keys.has(key)) {
        keys.set(key, {
          class_id: row.class_id,
          class_name: row.class_name,
          direction: row.direction,
          a: null,
          c: null,
        });
      }
      const entry = keys.get(key);
      if (rows === aCounts) entry.a = row;
      else entry.c = row;
    }
  };
  collect(aCounts);
  collect(cCounts);

  for (const entry of keys.values()) {
    const tr = document.createElement('tr');

    const tdClass = document.createElement('td');
    tdClass.textContent = entry.class_name || `类别 ${entry.class_id}`;
    tr.appendChild(tdClass);

    const tdDir = document.createElement('td');
    tdDir.textContent = entry.direction === 'A_to_B' ? 'A → B' : 'B → A';
    tr.appendChild(tdDir);

    const tdA = document.createElement('td');
    tdA.textContent = entry.a ? String(entry.a.predicted) : '—';
    tr.appendChild(tdA);

    const tdC = document.createElement('td');
    tdC.textContent = entry.c ? String(entry.c.predicted) : '—';
    tr.appendChild(tdC);

    const tdTruth = document.createElement('td');
    if (reviewed && entry.a && entry.a.true != null) {
      tdTruth.textContent = String(entry.a.true);
    } else {
      tdTruth.textContent = '—';
    }
    tr.appendChild(tdTruth);

    const tdStatus = document.createElement('td');
    if (reviewed) {
      const err = entry.a && entry.a.absolute_error != null ? entry.a.absolute_error : null;
      if (err != null) {
        tdStatus.textContent = `误差 ${err}`;
      } else if (reviewer) {
        tdStatus.textContent = `已复核 · ${reviewer}`;
      } else {
        tdStatus.textContent = '已复核';
      }
    } else {
      const span = document.createElement('span');
      span.className = 'pending-tag';
      span.textContent = '待人工复核';
      tdStatus.appendChild(span);
    }
    tr.appendChild(tdStatus);

    body.appendChild(tr);
  }
}

/* ------------------------------------------------------------------ */
/* Player                                                              */
/* ------------------------------------------------------------------ */

function setupPlayerForJob(job) {
  const video = state.videos.find((v) => v.video_id === job.video_id);
  const report = job && job.report ? job.report : null;
  const fps = (report && Number.isFinite(report.fps) && report.fps > 0)
    ? report.fps
    : (video && Number.isFinite(video.fps) && video.fps > 0 ? video.fps : null);
  const totalFrames = (report && Number.isFinite(report.frame_count) && report.frame_count > 0)
    ? report.frame_count
    : (report && Number.isFinite(report.frames) && report.frames > 0
      ? report.frames
      : (video && Number.isFinite(video.frame_count) ? video.frame_count : 0));

  show(el.blankA, false);
  show(el.blankC, false);
  el.frameA.hidden = true;
  el.canvasA.hidden = true;
  show(el.videoA, true);
  show(el.videoC, true);

  el.videoA.src = `${API}/jobs/${job.job_id}/files/A.mp4`;
  applyRightMode(job);

  if (!state.player) {
    state.player = createPlayer({
      left: el.videoA,
      right: el.videoC,
      playButton: el.btnPlay,
      prevButton: el.btnPrev,
      nextButton: el.btnNext,
      timeline: el.timeline,
      timeLabel: el.frameReadout,
      speedSelect: el.speedSelect,
    });
  }
  state.player.setMeta(fps, totalFrames);
  show(el.transport, true);
}

function applyRightMode(job) {
  const mode = state.rightMode;
  const file = mode === 'B' ? 'B.mp4' : 'C.mp4';
  el.videoC.src = `${API}/jobs/${job.job_id}/files/${file}`;
  setText(el.cardCTitle, mode === 'B' ? 'B 可切换' : 'C 改进版');
}

/* ------------------------------------------------------------------ */
/* Exports                                                             */
/* ------------------------------------------------------------------ */

function enableExports(enabled) {
  el.exportCsv.disabled = !enabled;
  el.exportReport.disabled = !enabled;
  el.exportA.disabled = !enabled;
  el.exportC.disabled = !enabled;
  el.exportDetections.disabled = !enabled;
}

function downloadFile(path, filename) {
  const a = document.createElement('a');
  a.href = path;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function exportFile(kind) {
  const job = state.job;
  if (!job) return;
  const base = `${API}/jobs/${job.job_id}/files/`;
  switch (kind) {
    case 'csv':
      downloadFile(`${base}counts.csv`, 'counts.csv');
      break;
    case 'report':
      downloadFile(`${base}report.json`, 'report.json');
      break;
    case 'a':
      downloadFile(`${base}A.mp4`, 'A.mp4');
      break;
    case 'c':
      downloadFile(`${base}C.mp4`, 'C.mp4');
      break;
    case 'detections':
      downloadFile(`${base}detections.jsonl`, 'detections.jsonl');
      break;
  }
}

/* ------------------------------------------------------------------ */
/* Tabs / review                                                       */
/* ------------------------------------------------------------------ */

function switchTab(tab) {
  if (tab === 'review') {
    const completed = Boolean(state.job && state.job.state === 'completed');
    if (!completed) {
      toast('任务完成后才能进入事件复核', 'error');
      return;
    }
    if (state.truth && state.truth.isDirty() && state.loadedJobId && state.loadedJobId !== state.job.job_id) {
      const ok = window.confirm('当前复核草稿尚未保存，确定切换任务并丢弃草稿吗？');
      if (!ok) return;
    }
  }
  state.activeTab = tab;
  const isReview = tab === 'review';
  el.tabCompare.classList.toggle('is-active', !isReview);
  el.tabReview.classList.toggle('is-active', isReview);
  el.tabCompare.setAttribute('aria-selected', String(!isReview));
  el.tabReview.setAttribute('aria-selected', String(isReview));

  show(el.compareView, !isReview);
  show(el.transport, !isReview && Boolean(state.job && state.job.state === 'completed'));
  show(document.querySelector('.table-section'), !isReview);
  show(document.querySelector('.exports'), !isReview);
  show(el.modeSelect.parentElement, !isReview);

  if (isReview) {
    openReview();
  } else {
    closeReview();
  }
}

function openReview() {
  const main = document.querySelector('.main');
  if (main && el.reviewView.parentElement !== main) {
    main.appendChild(el.reviewView);
  }
  show(el.reviewView, true);
  if (!state.truth) {
    state.truth = createTruth({
      root: el.reviewView,
      api: {
        getTruth: (id) => apiGet(`/jobs/${id}/truth`),
        putTruth: (id, body) => apiPut(`/jobs/${id}/truth`, body),
        getHealth: () => apiGet('/health'),
      },
      onSaved: () => {
        loadEvaluation(state.selectedJobId).then(renderCounts);
      },
    });
  }
  const job = state.job;
  if (!job || job.state !== 'completed') {
    return;
  }
  if (state.loadedJobId === job.job_id) {
    return;
  }
  if (state.truth.isDirty() && state.loadedJobId && state.loadedJobId !== job.job_id) {
    const ok = window.confirm('当前复核草稿尚未保存，确定切换任务并丢弃草稿吗？');
    if (!ok) return;
  }
  const video = state.videos.find((v) => v.video_id === job.video_id);
  const report = job.report || null;
  const fps = (report && Number.isFinite(report.fps) && report.fps > 0)
    ? report.fps
    : (video && Number.isFinite(video.fps) && video.fps > 0 ? video.fps : null);
  const totalFrames = (report && Number.isFinite(report.frame_count) && report.frame_count > 0)
    ? report.frame_count
    : (report && Number.isFinite(report.frames) && report.frames > 0
      ? report.frames
      : (video && Number.isFinite(video.frame_count) ? video.frame_count : 0));
  state.loadedJobId = job.job_id;
  state.truth.load(job.job_id, {
    fps,
    totalFrames,
    sourceUrl: `${API}/jobs/${job.job_id}/files/source.mp4`,
  });
}

function closeReview() {
  show(el.reviewView, false);
}

async function apiPut(path, body) {
  const res = await fetch(API + path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const data = await res.json();
      detail = data && data.error ? String(data.error) : '';
    } catch (_) {
      detail = '';
    }
    throw new Error(detail || `请求失败（${res.status}）`);
  }
  return res.json();
}

/* ------------------------------------------------------------------ */
/* New experiment                                                      */
/* ------------------------------------------------------------------ */

function newExperiment() {
  if (state.running) {
    toast('任务运行中，请先取消后再开始新实验', 'error');
    return;
  }
  if (state.truth && state.truth.isDirty()) {
    const ok = window.confirm('当前复核草稿尚未保存，确定开始新实验并丢弃草稿吗？');
    if (!ok) return;
  }
  state.frozen = false;
  state.job = null;
  state.evaluation = null;
  state.selectedJobId = null;
  state.loadedJobId = null;
  stopPolling();
  if (state.player) {
    state.player.destroy();
    state.player = null;
  }
  if (state.truth) {
    state.truth.destroy();
    state.truth = null;
  }
  enableExports(false);
  setEditorEnabled(true);
  show(el.transport, false);
  setProgress(0);
  show(el.progressWrap, false);
  setText(el.runStatus, '等待上传视频');
  renderCounts();
  if (state.selectedVideo) showFirstFrame(state.selectedVideo);
  else clearStages();
  updateReviewAvailability();
  toast('已开始新实验', 'ok');
}

/* ------------------------------------------------------------------ */
/* Event wiring                                                        */
/* ------------------------------------------------------------------ */

function wireEvents() {
  el.tabCompare.addEventListener('click', () => switchTab('compare'));
  el.tabReview.addEventListener('click', () => switchTab('review'));

  el.fileInput.addEventListener('change', (e) => {
    if (state.running) {
      toast('任务运行中，无法上传新视频', 'error');
      e.target.value = '';
      return;
    }
    const file = e.target.files && e.target.files[0];
    handleFile(file);
  });

  el.dropTarget.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.dropTarget.classList.add('is-drag');
  });
  el.dropTarget.addEventListener('dragleave', () => {
    el.dropTarget.classList.remove('is-drag');
  });
  el.dropTarget.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropTarget.classList.remove('is-drag');
    if (state.running) {
      toast('任务运行中，无法上传新视频', 'error');
      return;
    }
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
  });

  el.videoSelect.addEventListener('change', (e) => {
    if (state.running) {
      toast('任务运行中，无法切换视频', 'error');
      e.target.value = state.selectedVideo ? state.selectedVideo.video_id : '';
      return;
    }
    selectVideo(e.target.value);
  });

  el.btnDrawRoi.addEventListener('click', () => {
    if (guardFrozenGeometry()) return;
    if (state.editor) state.editor.setMode('roi');
  });
  el.btnDrawLine.addEventListener('click', () => {
    if (guardFrozenGeometry()) return;
    if (state.editor) state.editor.setMode('line');
  });
  el.btnResetGeo.addEventListener('click', () => {
    if (guardFrozenGeometry()) return;
    if (state.editor) state.editor.reset();
    updateGeoStatus();
  });

  el.btnRun.addEventListener('click', startRun);
  el.btnCancel.addEventListener('click', cancelRun);
  if (el.btnNewExperiment) {
    el.btnNewExperiment.addEventListener('click', newExperiment);
  }

  el.jobSelect.addEventListener('change', (e) => {
    const jobId = e.target.value;
    if (!jobId) return;
    if (state.running) {
      toast('任务运行中，无法切换历史任务', 'error');
      e.target.value = state.selectedJobId || '';
      return;
    }
    if (state.truth && state.truth.isDirty() && state.loadedJobId && state.loadedJobId !== jobId) {
      const ok = window.confirm('当前复核草稿尚未保存，确定切换任务并丢弃草稿吗？');
      if (!ok) {
        e.target.value = state.selectedJobId || '';
        return;
      }
    }
    state.selectedJobId = jobId;
    loadJobDetail(jobId);
  });

  el.modeSelect.addEventListener('change', (e) => {
    state.rightMode = e.target.value;
    if (state.job && state.job.state === 'completed') applyRightMode(state.job);
  });

  if (el.fpsOverride) {
    el.fpsOverride.addEventListener('input', () => {
      if (state.frozen) {
        toast('结果已冻结，请点击“新实验”后再修改参数', 'error');
      }
    });
  }

  el.exportCsv.addEventListener('click', () => exportFile('csv'));
  el.exportReport.addEventListener('click', () => exportFile('report'));
  el.exportA.addEventListener('click', () => exportFile('a'));
  el.exportC.addEventListener('click', () => exportFile('c'));
  el.exportDetections.addEventListener('click', () => exportFile('detections'));
}

async function loadJobDetail(jobId) {
  try {
    const job = await apiGet(`/jobs/${jobId}`);
    state.job = job;
    state.selectedVideo = state.videos.find(v => v.video_id === job.video_id) || null;
    el.videoSelect.value = state.selectedVideo?.video_id || '';
    renderVideoMeta();
    ensureEditor().setConfig(job.config);
    updateGeoStatus();
    if (job.state === 'completed') {
      state.frozen = true;
      setEditorEnabled(false);
      await loadEvaluation(jobId);
      renderCounts();
      enableExports(true);
      setupPlayerForJob(job);
      setProgress(100);
      setText(el.runStatus, '对比完成');
      updateReviewAvailability();
    } else if (job.state === 'running' || job.state === 'queued') {
      state.running = true;
      updateRunButton();
      show(el.btnCancel, true);
      show(el.progressWrap, true);
      startPolling(jobId);
    }
  } catch (err) {
    toast(`加载任务失败：${err.message}`, 'error');
  }
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

async function boot() {
  wireEvents();
  renderCounts();
  updateReviewAvailability();
  await loadHealth();
  await loadVideos();
  await loadJobs();
  updateRunButton();
}

boot();
