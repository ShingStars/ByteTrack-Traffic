// truth.js — manual ground-truth review panel.
//
// createTruth({ root, api, onSaved }) -> {
//   load(jobId, meta) -> Promise<void>   // meta: { fps, totalFrames, sourceUrl }
//   isDirty() -> boolean
//   destroy() -> void
// }
//
// root renders its own DOM. No predicted boxes/events are shown initially to
// avoid bias. Truth is never autofilled from predictions. Saving a draft does
// not mark the job reviewed; only the explicit checkbox + 保存复核 does.
//
// api contract:
//   GET  /api/jobs/:id/truth -> { events:[], reviewed:false, reviewer:'' }
//   PUT  /api/jobs/:id/truth -> { events:[{vehicle_id,class_id,direction,frame,uncertain}], reviewed, reviewer }
//   GET  /api/jobs/:id/evaluation -> { status, reviewer, truth_events, excluded_uncertain, modes:{A:{counts,events},B:{},C:{}} }

export function createTruth({ root, api, onSaved }) {
  let jobId = null;
  let meta = { fps: null, totalFrames: 0, sourceUrl: '' };
  let events = [];
  let reviewed = false;
  let reviewer = '';
  let dirty = false;
  let classes = [];
  let destroyed = false;

  root.innerHTML = '';

  const wrap = document.createElement('div');
  wrap.className = 'truth-panel';

  const video = document.createElement('video');
  video.className = 'truth-video';
  video.controls = true;
  video.preload = 'metadata';

  const frameRow = document.createElement('div');
  frameRow.className = 'truth-frame-row';
  const frameLabel = document.createElement('span');
  frameLabel.className = 'truth-frame-label';
  frameLabel.textContent = '当前帧 0';
  const stepBack = document.createElement('button');
  stepBack.type = 'button';
  stepBack.textContent = '上一帧';
  const stepFwd = document.createElement('button');
  stepFwd.type = 'button';
  stepFwd.textContent = '下一帧';
  frameRow.append(stepBack, stepFwd, frameLabel);

  const instructions = document.createElement('p');
  instructions.className = 'truth-instructions';
  instructions.textContent = '逐个记录清晰穿线事件；不确定事件不纳入评分。保存草稿不会标记已复核。';

  const form = document.createElement('div');
  form.className = 'truth-form';

  const idInput = document.createElement('input');
  idInput.type = 'text';
  idInput.placeholder = '物理车辆 ID';
  idInput.setAttribute('aria-label', '物理车辆 ID');

  const classSelect = document.createElement('select');
  classSelect.setAttribute('aria-label', '类别');

  const dirSelect = document.createElement('select');
  dirSelect.setAttribute('aria-label', '方向');
  [['A_to_B', 'A → B'], ['B_to_A', 'B → A']].forEach(([v, t]) => {
    const o = document.createElement('option');
    o.value = v; o.textContent = t;
    dirSelect.appendChild(o);
  });

  const frameInput = document.createElement('input');
  frameInput.type = 'number';
  frameInput.min = '0';
  frameInput.setAttribute('aria-label', '帧号');

  const uncertainWrap = document.createElement('label');
  uncertainWrap.className = 'truth-uncertain';
  const uncertainBox = document.createElement('input');
  uncertainBox.type = 'checkbox';
  uncertainWrap.append(uncertainBox, document.createTextNode('不确定'));

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'truth-add';
  addBtn.textContent = '添加事件';

  form.append(idInput, classSelect, dirSelect, frameInput, uncertainWrap, addBtn);

  const table = document.createElement('table');
  table.className = 'truth-table';
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  ['车辆 ID', '类别', '方向', '帧', '不确定', '操作'].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  const tbody = document.createElement('tbody');
  table.append(thead, tbody);

  const reviewRow = document.createElement('div');
  reviewRow.className = 'truth-review-row';
  const reviewerInput = document.createElement('input');
  reviewerInput.type = 'text';
  reviewerInput.placeholder = '复核人姓名';
  reviewerInput.setAttribute('aria-label', '复核人姓名');
  const confirmWrap = document.createElement('label');
  confirmWrap.className = 'truth-confirm';
  const confirmBox = document.createElement('input');
  confirmBox.type = 'checkbox';
  confirmWrap.append(confirmBox, document.createTextNode('我已逐帧核对本视频全部清晰穿线事件'));
  reviewRow.append(reviewerInput, confirmWrap);

  const btnRow = document.createElement('div');
  btnRow.className = 'truth-btn-row';
  const draftBtn = document.createElement('button');
  draftBtn.type = 'button';
  draftBtn.textContent = '保存草稿';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'truth-save';
  saveBtn.textContent = '保存复核';
  btnRow.append(draftBtn, saveBtn);

  const status = document.createElement('p');
  status.className = 'truth-status';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');

  wrap.append(video, frameRow, instructions, form, table, reviewRow, btnRow, status);
  root.appendChild(wrap);

  function setStatus(msg, isError) {
    status.textContent = msg || '';
    status.classList.toggle('truth-error', !!isError);
  }

  function markDirty() { dirty = true; }

  function fpsValue() {
    return Number.isFinite(meta.fps) && meta.fps > 0 ? meta.fps : 30;
  }

  function maxFrame() {
    const total = Number.isFinite(meta.totalFrames) && meta.totalFrames > 0 ? meta.totalFrames : 0;
    return total > 0 ? total - 1 : Infinity;
  }

  function currentFrame() {
    const max = maxFrame();
    const raw = Math.floor((video.currentTime || 0) * fpsValue() + 1e-3);
    if (!Number.isFinite(raw) || raw < 0) return 0;
    if (Number.isFinite(max)) return Math.min(raw, max);
    return raw;
  }

  function updateFrameLabel() {
    const frame = currentFrame();
    frameLabel.textContent = `当前帧 ${frame}`;
    frameInput.min = '0';
    const max = maxFrame();
    if (Number.isFinite(max)) {
      frameInput.max = String(max);
    } else {
      frameInput.removeAttribute('max');
    }
    frameInput.placeholder = '过线帧号';
    if (document.activeElement !== frameInput) {
      frameInput.value = String(frame);
    }
  }

  function renderClasses() {
    classSelect.innerHTML = '';
    classes.forEach(c => {
      const o = document.createElement('option');
      o.value = String(c.id);
      o.textContent = c.name;
      classSelect.appendChild(o);
    });
  }

  function renderTable() {
    tbody.innerHTML = '';
    events.forEach((ev, idx) => {
      const tr = document.createElement('tr');
      const cls = classes.find(c => c.id === ev.class_id);
      const cells = [
        ev.vehicle_id,
        cls ? cls.name : String(ev.class_id),
        ev.direction === 'A_to_B' ? 'A → B' : 'B → A',
        String(ev.frame),
        ev.uncertain ? '是' : '否'
      ];
      cells.forEach(v => {
        const td = document.createElement('td');
        td.textContent = v;
        tr.appendChild(td);
      });
      const td = document.createElement('td');
      const del = document.createElement('button');
      del.type = 'button';
      del.textContent = '删除';
      del.addEventListener('click', () => {
        events.splice(idx, 1);
        markDirty();
        renderTable();
      });
      td.appendChild(del);
      tr.appendChild(td);
      tr.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') return;
        video.currentTime = ev.frame / fpsValue();
      });
      tbody.appendChild(tr);
    });
  }

  function addEvent() {
    const vid = idInput.value.trim();
    if (!vid) { setStatus('请填写物理车辆 ID', true); return; }
    const direction = dirSelect.value;
    const duplicate = events.some(ev => ev.vehicle_id === vid && ev.direction === direction);
    if (duplicate) {
      setStatus('同一物理车辆在同一方向只能标注一次', true);
      return;
    }
    const blank = frameInput.value.trim() === '';
    const frame = blank ? currentFrame() : Number(frameInput.value);
    if (!Number.isInteger(frame)) { setStatus('帧号必须为整数', true); return; }
    const max = maxFrame();
    if (frame < 0 || (Number.isFinite(max) && frame > max)) {
      setStatus('帧号超出范围', true);
      return;
    }
    events.push({
      vehicle_id: vid,
      class_id: Number(classSelect.value),
      direction,
      frame,
      uncertain: uncertainBox.checked
    });
    idInput.value = '';
    uncertainBox.checked = false;
    markDirty();
    renderTable();
    setStatus('已添加事件（未保存）');
  }

  async function save(isReviewed) {
    if (isReviewed) {
      if (!reviewerInput.value.trim()) { setStatus('复核人姓名必填', true); return; }
      if (!confirmBox.checked) { setStatus('请勾选已逐帧核对', true); return; }
    }
    const payload = {
      events: events.map(e => ({
        vehicle_id: e.vehicle_id,
        class_id: e.class_id,
        direction: e.direction,
        frame: e.frame,
        uncertain: !!e.uncertain
      })),
      reviewed: !!isReviewed,
      reviewer: reviewerInput.value.trim()
    };
    try {
      const res = await api.putTruth(jobId, payload);
      const status = res && res.status ? String(res.status) : '';
      reviewed = status === 'reviewed';
      reviewer = (res && res.reviewer) ? String(res.reviewer) : '';
      if (reviewed) {
        reviewerInput.value = reviewer;
        confirmBox.checked = true;
      }
      dirty = false;
      setStatus(isReviewed ? '已保存复核' : '草稿已保存');
      if (typeof onSaved === 'function') onSaved(res);
    } catch (err) {
      setStatus('保存失败：' + (err && err.message ? err.message : '未知错误'), true);
    }
  }

  addBtn.addEventListener('click', addEvent);
  draftBtn.addEventListener('click', () => save(false));
  saveBtn.addEventListener('click', () => save(true));
  stepBack.addEventListener('click', () => {
    video.currentTime = Math.max(0, (currentFrame() - 1) / fpsValue());
  });
  stepFwd.addEventListener('click', () => {
    const max = maxFrame();
    const next = currentFrame() + 1;
    video.currentTime = Math.min(max, next) / fpsValue();
  });
  video.addEventListener('timeupdate', updateFrameLabel);
  video.addEventListener('seeked', updateFrameLabel);
  reviewerInput.addEventListener('input', markDirty);
  confirmBox.addEventListener('change', markDirty);

  async function load(id, m) {
    if (!id) {
      jobId = null;
      meta = { fps: null, totalFrames: 0, sourceUrl: '' };
      dirty = false;
      reviewed = false;
      reviewer = '';
      events = [];
      confirmBox.checked = false;
      reviewerInput.value = '';
      tbody.innerHTML = '';
      setStatus('');
      video.pause();
      video.removeAttribute('src');
      video.load();
      return;
    }
    jobId = id;
    meta = Object.assign({ fps: null, totalFrames: 0, sourceUrl: '' }, m || {});
    dirty = false;
    reviewed = false;
    reviewer = '';
    events = [];
    confirmBox.checked = false;
    reviewerInput.value = '';
    setStatus('');
    if (meta.sourceUrl) video.src = meta.sourceUrl;
    try {
      const truth = await api.getTruth(jobId);
      if (destroyed) return;
      if (!classes.length && typeof api.getHealth === 'function') {
        try {
          const health = await api.getHealth();
          if (health && Array.isArray(health.classes)) classes = health.classes;
        } catch (_) { /* health optional */ }
      }
      renderClasses();
      events = Array.isArray(truth.events) ? truth.events.slice() : [];
      reviewed = truth.status === 'reviewed' || truth.reviewed === true;
      reviewer = truth.reviewer || '';
      reviewerInput.value = reviewer;
      if (reviewed) confirmBox.checked = true;
      renderTable();
      updateFrameLabel();
      if (reviewed) setStatus('该任务已复核');
    } catch (err) {
      setStatus('加载失败：' + (err && err.message ? err.message : '未知错误'), true);
    }
  }

  function isDirty() { return dirty; }

  function destroy() {
    destroyed = true;
    video.pause();
    video.removeAttribute('src');
    video.load();
    root.innerHTML = '';
  }

  function setClasses(list) {
    if (Array.isArray(list) && list.length) {
      classes = list;
      renderClasses();
    }
  }

  return { load, isDirty, destroy, setClasses };
}
