/* Player controller: dual-video sync, frame stepping, timeline, speed. */

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function finitePositive(v) {
    return typeof v === 'number' && isFinite(v) && v > 0;
  }

export function createPlayer(refs) {
    var left = refs.left;
    var right = refs.right;
    var playButton = refs.playButton;
    var prevButton = refs.prevButton;
    var nextButton = refs.nextButton;
    var timeline = refs.timeline;
    var timeLabel = refs.timeLabel;
    var speedSelect = refs.speedSelect;

    var fps = 30;
    var totalFrames = 0;
    var frameIndex = 0;
    var playing = false;
    var destroyed = false;
    var listeners = [];

    function on(target, type, handler, opts) {
      if (!target) return;
      target.addEventListener(type, handler, opts);
      listeners.push([target, type, handler, opts]);
    }

    function mediaReady(video) {
      if (!video) return false;
      if (video.readyState < 1) return false;
      return finitePositive(video.duration);
    }

    function bothReady() {
      return mediaReady(left) && mediaReady(right);
    }

    function syncIndicator() {
      var el = document.getElementById('sync-indicator');
      if (!el) return;
      if (!bothReady()) {
        el.textContent = '等待视频';
        return;
      }
      if (!playing) {
        el.textContent = '同步';
        return;
      }
      var gap = Math.abs((left.currentTime || 0) - (right.currentTime || 0));
      el.textContent = gap > 1 / Math.max(fps, 1) ? '校准中' : '同步';
    }

    function updateControls() {
      var ready = bothReady();
      if (playButton) playButton.disabled = !ready;
      if (prevButton) prevButton.disabled = !ready;
      if (nextButton) nextButton.disabled = !ready;
      if (timeline) timeline.disabled = !ready;
      if (speedSelect) speedSelect.disabled = !ready;
      syncIndicator();
    }

    function updateLabel() {
      if (!timeLabel) return;
      var n = totalFrames > 0 ? totalFrames : 0;
      var f = n > 0 ? clamp(frameIndex, 0, n - 1) : 0;
      timeLabel.textContent = '第 ' + f + ' 帧 · 共 ' + n + ' 帧';
    }

    function updatePlayButton() {
      if (!playButton) return;
      playButton.textContent = playing ? 'Ⅱ' : '▶';
      playButton.setAttribute('aria-label', playing ? '暂停' : '播放');
    }

    function currentFrameFrom(video) {
      if (!video || !finitePositive(video.duration)) return 0;
      var f = Math.floor((video.currentTime || 0) * fps + 1e-3);
      var n = totalFrames > 0 ? totalFrames : Math.max(1, Math.floor(video.duration * fps));
      return clamp(f, 0, n - 1);
    }

    function seekToFrame(f) {
      if (!bothReady()) return;
      var n = totalFrames > 0 ? totalFrames : Math.max(1, Math.floor(Math.min(left.duration, right.duration) * fps));
      var target = clamp(Math.floor(f), 0, n - 1);
      frameIndex = target;
      var t = target / fps;
      try { left.currentTime = t; } catch (e) {}
      try { right.currentTime = t; } catch (e) {}
      if (timeline) timeline.value = String(n > 1 ? Math.round((target / (n - 1)) * 1000) : 0);
      updateLabel();
      syncIndicator();
    }

    function pauseBoth() {
      try { left.pause(); } catch (e) {}
      try { right.pause(); } catch (e) {}
      playing = false;
      updatePlayButton();
      syncIndicator();
    }

    function playBoth() {
      if (!bothReady()) return;
      var p1 = left.play();
      var p2 = right.play();
      playing = true;
      updatePlayButton();
      Promise.all([p1, p2]).catch(function (err) {
        pauseBoth();
        var el = document.getElementById('sync-indicator');
        if (el) el.textContent = '播放失败';
        if (window.console && console.error) console.error('playback failed', err);
      });
    }

    function togglePlay() {
      if (playing) pauseBoth();
      else playBoth();
    }

    function step(delta) {
      if (!bothReady()) return;
      pauseBoth();
      seekToFrame(frameIndex + delta);
    }

    function applySpeed() {
      var v = speedSelect ? parseFloat(speedSelect.value) : 1;
      if (!isFinite(v) || v <= 0) v = 1;
      try { left.playbackRate = v; } catch (e) {}
      try { right.playbackRate = v; } catch (e) {}
    }

    function alignRightToLeft() {
      if (!bothReady()) return;
      var gap = Math.abs((left.currentTime || 0) - (right.currentTime || 0));
      if (gap > 1 / Math.max(fps, 1)) {
        try { right.currentTime = left.currentTime; } catch (e) {}
      }
    }

    function onLeftLoadedMetadata() {
      if (mediaReady(left) && mediaReady(right)) {
        try { right.currentTime = clamp(left.currentTime || 0, 0, Math.max(0, right.duration - 1e-3)); } catch (e) {}
      }
      updateControls();
    }

    function onRightLoadedMetadata() {
      if (mediaReady(left) && mediaReady(right)) {
        try { right.currentTime = clamp(left.currentTime || 0, 0, Math.max(0, right.duration - 1e-3)); } catch (e) {}
      }
      updateControls();
    }

    function onTimeUpdate() {
      if (!bothReady()) return;
      if (playing) alignRightToLeft();
      frameIndex = currentFrameFrom(left);
      if (timeline && totalFrames > 1) {
        timeline.value = String(Math.round((frameIndex / (totalFrames - 1)) * 1000));
      }
      updateLabel();
      syncIndicator();
    }

    function onSeeked() {
      frameIndex = currentFrameFrom(left);
      updateLabel();
      syncIndicator();
    }

    function onEnded() {
      pauseBoth();
      seekToFrame((totalFrames > 0 ? totalFrames : 1) - 1);
    }

    function onTimelineInput() {
      if (!bothReady() || !timeline) return;
      var n = totalFrames > 0 ? totalFrames : 1;
      var f = n > 1 ? Math.round((parseFloat(timeline.value) / 1000) * (n - 1)) : 0;
      pauseBoth();
      seekToFrame(f);
    }

    function onSpeedChange() {
      applySpeed();
    }

    function onEmptied() {
      pauseBoth();
      updateControls();
    }

    function onLoadedCanPlay() {
      updateControls();
    }

    function setMeta(fpsValue, totalFramesValue) {
      var meta = { fps: Number(fpsValue), totalFrames: Number(totalFramesValue) };
      if (finitePositive(meta.fps)) fps = meta.fps;
      if (typeof meta.totalFrames === 'number' && isFinite(meta.totalFrames) && meta.totalFrames > 0) {
        totalFrames = Math.floor(meta.totalFrames);
      } else if (mediaReady(left)) {
        totalFrames = Math.max(1, Math.floor(left.duration * fps));
      } else if (mediaReady(right)) {
        totalFrames = Math.max(1, Math.floor(right.duration * fps));
      } else {
        totalFrames = 0;
      }
      frameIndex = clamp(frameIndex, 0, Math.max(0, totalFrames - 1));
      if (timeline) timeline.max = '1000';
      updateLabel();
      updateControls();
    }

    function destroy() {
      if (destroyed) return;
      destroyed = true;
      pauseBoth();
      for (var i = 0; i < listeners.length; i++) {
        var l = listeners[i];
        try { l[0].removeEventListener(l[1], l[2], l[3]); } catch (e) {}
      }
      listeners = [];
    }

    on(left, 'loadedmetadata', onLeftLoadedMetadata);
    on(right, 'loadedmetadata', onRightLoadedMetadata);
    on(left, 'loadedmetadata', onLoadedCanPlay);
    on(right, 'loadedmetadata', onLoadedCanPlay);
    on(left, 'canplay', onLoadedCanPlay);
    on(right, 'canplay', onLoadedCanPlay);
    on(left, 'timeupdate', onTimeUpdate);
    on(right, 'timeupdate', onTimeUpdate);
    on(left, 'seeked', onSeeked);
    on(right, 'seeked', onSeeked);
    on(left, 'ended', onEnded);
    on(right, 'ended', onEnded);
    on(left, 'emptied', onEmptied);
    on(right, 'emptied', onEmptied);
    on(playButton, 'click', togglePlay);
    on(prevButton, 'click', function () { step(-1); });
    on(nextButton, 'click', function () { step(1); });
    on(timeline, 'input', onTimelineInput);
    on(speedSelect, 'change', onSpeedChange);

    applySpeed();
    updatePlayButton();
    updateControls();
    updateLabel();

    return {
      setMeta: setMeta,
      destroy: destroy
    };
  }
