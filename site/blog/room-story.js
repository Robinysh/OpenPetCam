/* Small, independent illustrations: static without JS; motion only while in view. */
(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const saveData = navigator.connection?.saveData;
  const illustrations = Array.from(document.querySelectorAll('[data-motion]'), figure => {
    const button = figure.querySelector('.motion-toggle');
    const video = figure.querySelector('video');
    const state = { figure, button, video, visible: false, requested: !reducedMotion.matches && !saveData };

    state.update = () => {
      const playing = state.requested && state.visible && !document.hidden;
      figure.dataset.playing = String(playing);
      button.textContent = state.requested ? 'Pause animation' : 'Play animation';
      button.setAttribute('aria-label', `${state.requested ? 'Pause' : 'Play'}: ${figure.dataset.motion}`);
      if (!video) return;
      if (!playing) {
        video.pause();
        return;
      }
      if (!video.getAttribute('src')) video.src = video.dataset.src;
      if (video.paused) video.play().catch(error => {
        // Scrolling away can interrupt a pending play. Preserve the reader's intent.
        if (error.name === 'AbortError') return;
        state.requested = false;
        state.update();
      });
    };
    button.hidden = false;
    button.addEventListener('click', () => {
      state.requested = !state.requested;
      state.update();
    });
    state.update();
    return state;
  });

  const byFigure = new Map(illustrations.map(state => [state.figure, state]));
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        const state = byFigure.get(entry.target);
        state.visible = entry.isIntersecting && entry.intersectionRatio >= 0.25;
        state.update();
      });
    }, { threshold: [0, 0.25] });
    illustrations.forEach(state => observer.observe(state.figure));
  } else {
    illustrations.forEach(state => { state.visible = true; state.update(); });
  }

  document.addEventListener('visibilitychange', () => illustrations.forEach(state => state.update()));
  reducedMotion.addEventListener('change', () => {
    illustrations.forEach(state => {
      if (reducedMotion.matches) state.requested = false;
      state.update();
    });
  });
})();
