/* Original film scenes, kept separate from the article's schematic illustrations. */
(() => {
  'use strict';

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const connection = navigator.connection;
  const players = [];
  const timeLabel = seconds => {
    const tenths = Math.max(0, Math.round((Number.isFinite(seconds) ? seconds : 0) * 10));
    return `${Math.floor(tenths / 600)}:${String(Math.floor(tenths / 10) % 60).padStart(2, '0')}.${tenths % 10}`;
  };

  document.querySelectorAll('[data-film]').forEach(figure => {
    const video = figure.querySelector('video');
    const stage = figure.querySelector('.film-stage');
    const chapters = Array.from(figure.querySelectorAll('[data-film-src]'));
    const play = figure.querySelector('[data-film-play]');
    const scrub = figure.querySelector('[data-film-scrub]');
    const clock = figure.querySelector('[data-film-time]');
    const status = figure.querySelector('.film-status');
    if (!video || !stage || !chapters.length || !play || !scrub || !clock) return;
    video.muted = true;
    video.loop = false;

    let selected = Math.max(0, chapters.findIndex(button => button.getAttribute('aria-pressed') === 'true'));
    let generation = 0, playToken = 0, loaded = false, requested = false;
    let ended = false, failed = false, autoAttempted = false, visible = false;
    let pendingSeek = null, seekFrame = 0, seekAttempts = 0;
    const duration = () => Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const frameSeconds = 1 / (Number(video.dataset.filmFps) || 24);
    const title = () => chapters[selected].dataset.filmTitle || chapters[selected].textContent.trim();
    const say = message => { if (status && status.textContent !== message) status.textContent = message; };
    const currentSource = () => {
      const expected = new URL(chapters[selected].dataset.filmSrc, document.baseURI).href;
      return loaded && video.currentSrc === expected;
    };

    function update() {
      const total = duration();
      const current = pendingSeek == null ? (ended ? total : Math.max(0, video.currentTime || 0)) : pendingSeek * total;
      scrub.value = pendingSeek == null ? (total ? String(Math.round(current / total * 1000)) : '0') : String(Math.round(pendingSeek * 1000));
      clock.textContent = `${timeLabel(current)} / ${total ? timeLabel(total) : '—'}`;
      scrub.setAttribute('aria-valuetext', total ? `${timeLabel(current)} of ${timeLabel(total)}` :
        pendingSeek == null ? 'Clip time; duration loading' : `${Math.round(pendingSeek * 100)}% of clip; duration loading`);
      play.textContent = requested ? 'Pause' : ended ? 'Replay' : failed ? 'Retry' : 'Play';
      play.setAttribute('aria-label', `${requested ? 'Pause' : ended ? 'Replay' : failed ? 'Retry' : 'Play'}: ${title()}`);
      play.setAttribute('aria-pressed', String(requested));
      figure.dataset.playing = String(requested && !video.paused);
    }

    function pause(message) {
      ++playToken;
      requested = false;
      video.pause();
      if (message) say(message);
      update();
    }

    function load() {
      if (loaded) return;
      failed = false;
      loaded = true;
      video.preload = 'auto';
      video.src = chapters[selected].dataset.filmSrc;
      say(`Loading ${title()}…`);
      video.load();
    }

    function applySeek() {
      seekFrame = 0;
      // Some decoders reset a metadata-only seek when their first frame arrives.
      // Keep the requested thumb until a decoded frame acknowledges that position.
      if (pendingSeek == null || !duration() || !currentSource() || video.readyState < 2) return;
      if (video.seeking) return;
      const value = pendingSeek;
      // EOF is beyond the last picture and some decoders jump back to zero there.
      // End selects the final authored frame; the UI reports the clip as complete.
      const target = Math.min(value * duration(), Math.max(0, duration() - frameSeconds));
      if (Math.abs(video.currentTime - target) <= 1 / 48) {
        pendingSeek = null;
        seekAttempts = 0;
        ended = value >= 1;
        update();
        return;
      }
      if (seekAttempts >= 3) {
        pendingSeek = null;
        ended = false;
        say('That point could not be reached. Try another position or play the scene.');
        update();
        return;
      }
      try {
        ++seekAttempts;
        video.currentTime = target;
        update();
      } catch { /* Keep the latest requested position until metadata is usable. */ }
    }

    function requestPlay() {
      const thisGeneration = generation, thisPlay = ++playToken;
      requested = true;
      load();
      if (ended) {
        pendingSeek = 0;
        seekAttempts = 0;
        ended = false;
        applySeek();
      }
      update();
      let attempt;
      try { attempt = video.play(); } catch (error) { attempt = Promise.reject(error); }
      Promise.resolve(attempt).then(() => {
        if (thisGeneration !== generation || thisPlay !== playToken) return;
        if (!requested || document.hidden || ('IntersectionObserver' in window && !visible)) {
          pause();
          return;
        }
        say('');
        update();
      }).catch(error => {
        if (thisGeneration !== generation || thisPlay !== playToken) return;
        requested = false;
        say(error.name === 'AbortError' ? 'Playback paused. Press Play to continue.' : 'Playback did not start. Press Play to try again.');
        update();
      });
    }

    function select(index, userInitiated = false) {
      ++generation;
      pause();
      selected = index;
      loaded = false;
      ended = false;
      failed = false;
      pendingSeek = null;
      seekAttempts = 0;
      if (seekFrame) cancelAnimationFrame(seekFrame);
      seekFrame = 0;
      video.removeAttribute('src');
      video.poster = chapters[index].dataset.filmPoster || '';
      video.setAttribute('aria-label', title());
      video.load();
      chapters.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
      scrub.value = '0';
      say(userInitiated ? `${title()}. Press Play or scrub to explore this scene.` : '');
      update();
    }

    play.addEventListener('click', () => {
      autoAttempted = true;
      if (requested) pause('Paused.');
      else requestPlay();
    });
    chapters.forEach((button, index) => button.addEventListener('click', () => {
      autoAttempted = true;
      if (index !== selected) select(index, true);
    }));
    scrub.addEventListener('input', () => {
      const position = Math.max(0, Math.min(1, Number(scrub.value) / 1000));
      autoAttempted = true;
      pause();
      pendingSeek = position;
      seekAttempts = 0;
      ended = false;
      load();
      if (!seekFrame) seekFrame = requestAnimationFrame(applySeek);
      say('Paused at the selected clip time.');
      update();
    });

    video.addEventListener('loadedmetadata', () => {
      if (!currentSource() || !duration()) return;
      applySeek();
      if (!requested) say('Ready. Press Play or scrub to explore this scene.');
      update();
    });
    video.addEventListener('loadeddata', () => {
      if (!currentSource()) return;
      applySeek();
      if (!requested && pendingSeek == null && seekAttempts < 3) say('');
    });
    video.addEventListener('canplay', applySeek);
    video.addEventListener('seeked', applySeek);
    video.addEventListener('timeupdate', update);
    video.addEventListener('seeking', update);
    video.addEventListener('playing', () => {
      if (!requested || document.hidden) { video.pause(); return; }
      say('');
      update();
    });
    video.addEventListener('waiting', () => { if (requested) say('Loading the scene…'); });
    video.addEventListener('ended', () => {
      if (!currentSource() || !video.ended) return;
      if (pendingSeek != null) { applySeek(); if (pendingSeek != null) return; }
      ++playToken;
      requested = false;
      ended = true;
      say('End of scene. Replay it or choose another view.');
      update();
    });
    video.addEventListener('error', () => {
      if (!loaded || !video.error) return;
      ++playToken;
      requested = false;
      loaded = false;
      failed = true;
      pendingSeek = null;
      video.removeAttribute('src');
      video.load(); // Restore the poster rather than leaving a broken media surface.
      say('This scene could not load. The still image remains; press Retry to try again.');
      update();
    });

    select(selected);
    figure.querySelector('.film-controls')?.removeAttribute('hidden');
    const chapterStrip = figure.querySelector('.film-chapters');
    if (chapterStrip) chapterStrip.hidden = chapters.length < 2;
    const player = {
      stage,
      pause,
      enter(inView, autoInView) {
        visible = inView;
        if (!inView) {
          if (requested) pause('Paused while this scene is out of view.');
          return;
        }
        if (!autoInView) return;
        if (autoAttempted) return;
        autoAttempted = true;
        if (!reducedMotion.matches && !connection?.saveData && !document.hidden) requestPlay();
      }
    };
    players.push(player);
  });

  if ('IntersectionObserver' in window) {
    const byStage = new Map(players.map(player => [player.stage, player]));
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => byStage.get(entry.target)?.enter(entry.isIntersecting && entry.intersectionRatio > 0, entry.intersectionRatio >= 0.25));
    }, { threshold: [0, 0.25] });
    players.forEach(player => observer.observe(player.stage));
  }
  // Without an observer, leave the poster and manual controls available; do not guess visibility.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) players.forEach(player => player.pause());
  });
  const pauseForPreference = () => {
    if (reducedMotion.matches || connection?.saveData) players.forEach(player => player.pause());
  };
  reducedMotion.addEventListener('change', pauseForPreference);
  connection?.addEventListener?.('change', pauseForPreference);
})();
