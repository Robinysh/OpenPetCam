/* A wholly invented room/day illustration. No household data, assets or APIs. */
(() => {
  'use strict';
  const PATCHES = [
    {id: 'window', name: 'By the window', x: 190, y: 115, w: 152, h: 145,
      minutes: [0,0,0,1,1,2,5,11,17,15,12,9,7,8,10,13,14,11,7,4,1,0,0,0],
      sightings: [0,0,0,0,0,0,0,1,2,1,0,0,0,1,1,2,2,1,0,0,0,0,0,0]},
    {id: 'rug', name: 'On the rug', x: 365, y: 280, w: 210, h: 150,
      minutes: [1,0,0,0,0,1,5,9,12,7,4,4,6,8,8,6,11,15,18,13,8,4,2,1],
      sightings: [0,0,0,0,0,0,1,1,2,1,0,0,1,1,1,1,2,3,3,2,1,0,0,0]},
    {id: 'sofa', name: 'Beside the sofa', x: 245, y: 300, w: 105, h: 145,
      minutes: [3,1,0,0,0,0,3,4,5,5,7,11,14,18,19,15,10,8,6,7,8,7,6,4],
      sightings: [0,0,0,0,0,0,0,1,1,0,1,1,2,3,3,2,1,1,1,1,1,1,0,0]},
    {id: 'hallway', name: 'In the hallway', x: 618, y: 335, w: 150, h: 145,
      minutes: [0,0,0,0,0,0,1,2,3,2,1,1,0,1,1,2,2,1,3,2,1,0,0,0],
      sightings: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
  ];
  const clampHour = value => Math.max(0, Math.min(24, Number(value) || 0));
  const clock = value => {
    const minutes = Math.round(clampHour(value) * 60);
    return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
  };
  const observations = patch => patch.minutes.flatMap((minutes, hour) => {
    if (!minutes) return [];
    const start = hour + (7 + (hour * 7 + PATCHES.indexOf(patch) * 11) % 22) / 60;
    const end = start + minutes / 60;
    return [{start, end, events: Array.from({length: patch.sightings[hour]}, (_, i) =>
      ({at: start + (i + 1) * (end - start) / (patch.sightings[hour] + 1), index: hour * 4 + i}))}];
  });
  const sample = (id, value) => {
    const patch = PATCHES.find(p => p.id === id);
    if (!patch) throw new Error('Unknown illustration patch');
    const hour = clampHour(value), start = Math.max(0, hour - 1);
    const intervals = observations(patch);
    const minutes = intervals.reduce((sum, q) => sum + Math.max(0, Math.min(hour, q.end) - Math.max(start, q.start)) * 60, 0);
    const events = intervals.flatMap(q => q.events).filter(q => q.at > start && q.at <= hour);
    return {patch, hour, start, minutes: Math.round(minutes * 10) / 10, events};
  };
  // Pure helpers are available to local Node contracts; no browser global is added.
  if (typeof module !== 'undefined' && module.exports) module.exports = {PATCHES, clock, sample, clampHour};
  if (typeof document === 'undefined') return;
  const root = document.querySelector('[data-room-day]');
  if (!root || root.dataset.dayReady === 'true') return;
  const scene = root.querySelector('[data-day-scene]');
  const controls = root.querySelector('[data-day-controls]');
  if (!scene || !controls) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const state = {hour: 18, patch: 'rug', appearance: true, sightings: true, coverage: false, playing: false, visible: false};
  const svgNS = 'http://www.w3.org/2000/svg';
  const node = (tag, attributes = {}, text) => {
    const el = document.createElement(tag);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const svgNode = (tag, attributes) => {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  };
  scene.innerHTML = `<svg viewBox="0 0 840 560" role="img" aria-labelledby="room-day-title room-day-desc">
    <title id="room-day-title">A sample room through one day</title>
    <desc id="room-day-desc">An invented room with four selectable floor patches. Use the named buttons below the drawing to inspect sample sightings and camera observation time. Sparse observation is not evidence of absence.</desc>
    <defs><clipPath id="room-day-floor"><path d="M100 80H610V285H790V500H100Z"/></clipPath>
      <pattern id="room-day-hatch" width="12" height="12" patternUnits="userSpaceOnUse"><path d="M-3 3L3-3M0 12L12 0M9 15L15 9" class="room-day__hatch-line"/></pattern>
      <linearGradient id="room-day-sun" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#e5ad55" stop-opacity=".6"/><stop offset="1" stop-color="#e5ad55" stop-opacity="0"/></linearGradient>
    </defs>
    <path class="room-day__floor" d="M100 80H610V285H790V500H100Z"/>
    <g clip-path="url(#room-day-floor)" class="room-day__ambience">
      <path class="room-day__sun" d="M147 84H515L652 494H206Z" fill="url(#room-day-sun)"/>
      <rect class="room-day__night" x="100" y="80" width="690" height="420"/>
    </g>
    <g class="room-day__zones"></g>
    <g class="room-day__furniture" aria-hidden="true">
      <path class="room-day__wall" d="M100 80H610V285H790V500H100Z"/>
      <path class="room-day__window" d="M145 80H515M145 88H515M238 80V88M330 80V88M423 80V88"/>
      <rect class="room-day__sofa" x="111" y="288" width="122" height="166" rx="14"/>
      <path class="room-day__line" d="M135 302V438M137 366H219M126 289V278M216 289V278M126 454V466M216 454V466"/>
      <rect class="room-day__rug" x="370" y="287" width="203" height="139" rx="7"/>
      <path class="room-day__rug-lines" d="M383 304H560M383 321H560M383 338H560M383 355H560M383 372H560M383 389H560M383 406H560"/>
      <ellipse class="room-day__table" cx="470" cy="186" rx="65" ry="37"/>
      <path class="room-day__line" d="M430 214L424 237M508 215L514 237"/>
      <path class="room-day__plant" d="M141 235H177L170 270H148ZM159 235V180M159 214Q130 213 133 191Q155 190 159 214M159 203Q181 199 184 176Q162 176 159 203"/>
      <path class="room-day__door" d="M659 499V444A55 55 0 0 1 714 499"/>
      <g class="room-day__camera"><rect x="570" y="103" width="30" height="23" rx="6"/><circle cx="585" cy="115" r="5"/><path d="M585 127V138M576 139H594"/></g>
    </g><g class="room-day__dots" aria-hidden="true"></g>
    <text class="room-day__map-note" x="100" y="539">A small, imagined room</text>
  </svg>`;
  const zones = scene.querySelector('.room-day__zones');
  const dots = scene.querySelector('.room-day__dots');
  const zoneNodes = new Map();
  PATCHES.forEach(patch => {
    const g = svgNode('g', {'data-day-patch': patch.id, class: 'room-day__zone', 'aria-hidden': 'true'});
    const rect = {x: patch.x, y: patch.y, width: patch.w, height: patch.h, rx: 8};
    const coverage = svgNode('rect', {...rect, class: 'room-day__coverage'});
    const hatch = svgNode('rect', {...rect, class: 'room-day__sparse', fill: 'url(#room-day-hatch)'});
    const hit = svgNode('rect', {...rect, class: 'room-day__hit'});
    g.append(coverage, hatch, hit); zones.append(g);
    zoneNodes.set(patch.id, {g, coverage, hatch});
  });
  const label = node('p', {class: 'room-day__label'}, 'Illustration · sample room and sightings');
  const timeRow = node('div', {class: 'room-day__time'});
  const timeLabel = node('label', {for: 'room-day-hour'}, 'Time in the sample day');
  const timeOutput = node('output', {for: 'room-day-hour'});
  timeRow.append(timeLabel, timeOutput);
  const range = node('input', {id: 'room-day-hour', type: 'range', min: 0, max: 24, step: .25, value: 18, 'aria-describedby': 'room-day-window'});
  const transport = node('div', {class: 'room-day__transport'});
  const presets = node('div', {class: 'room-day__presets', role: 'group', 'aria-label': 'Choose a time of day'});
  const presetNodes = [];
  [['Dawn', 6.5], ['Day', 12], ['Evening', 18], ['Night', 22]].forEach(([name, hour]) => {
    const button = node('button', {type: 'button', 'aria-pressed': 'false'}, name);
    button.addEventListener('click', () => { stop(); state.hour = hour; update(true); });
    presets.append(button); presetNodes.push({button, hour});
  });
  const play = node('button', {type: 'button', class: 'room-day__play', 'aria-pressed': 'false'}, 'Play day');
  transport.append(presets, play);
  const layers = node('div', {class: 'room-day__layers', role: 'group', 'aria-label': 'Illustration layers'});
  [['appearance', 'Daylight'], ['sightings', 'Sightings'], ['coverage', 'Camera coverage']].forEach(([key, title]) => {
    const toggleLabel = node('label');
    const input = node('input', {type: 'checkbox'}); input.checked = state[key];
    input.addEventListener('change', () => { state[key] = input.checked; update(true); });
    toggleLabel.append(input, document.createTextNode(title)); layers.append(toggleLabel);
  });
  const choices = node('div', {class: 'room-day__patches', role: 'group', 'aria-label': 'Choose a floor patch'});
  const patchButtons = new Map();
  PATCHES.forEach(patch => {
    const button = node('button', {type: 'button', 'aria-pressed': 'false'}, patch.name);
    button.addEventListener('click', () => { state.patch = patch.id; update(true); });
    choices.append(button); patchButtons.set(patch.id, button);
  });
  const detail = node('div', {class: 'room-day__detail'});
  const detailHeading = node('p', {class: 'room-day__detail-title'});
  const windowLabel = node('p', {id: 'room-day-window', class: 'room-day__window-label'});
  const counts = node('p', {class: 'room-day__counts'});
  const context = node('p', {class: 'room-day__context'});
  const motionNote = node('p', {class: 'room-day__motion-note'});
  const live = node('p', {class: 'room-day__sr', 'aria-live': 'polite', 'aria-atomic': 'true'});
  detail.append(detailHeading, counts, context);
  const layout = node('div', {class: 'room-day__layout'});
  const inspector = node('div', {class: 'room-day__inspector'});
  inspector.append(layers, choices, windowLabel, detail, motionNote);
  root.insertBefore(controls, scene);
  root.insertBefore(layout, scene);
  layout.append(scene, inspector);
  controls.append(label, timeRow, range, transport, live);
  let request = null, previous = 0, lastPaint = 0;
  function update(announce = false) {
    // Keep the clock, native slider value and evidence window on the same quarter-hour.
    const hour = Math.round(state.hour * 4) / 4;
    const active = sample(state.patch, hour);
    root.dataset.coverage = String(state.coverage); root.dataset.sightings = String(state.sightings);
    range.value = String(hour); timeOutput.value = clock(hour); timeOutput.textContent = clock(hour);
    range.setAttribute('aria-valuetext', clock(hour));
    windowLabel.textContent = active.hour === 0 ? 'Start of the sample day' :
      `${clock(active.start)}–${clock(active.hour)} · sample window`;
    detailHeading.textContent = active.patch.name;
    counts.textContent = `${active.events.length} ${active.events.length === 1 ? 'sighting' : 'sightings'} · ${active.minutes} ${active.minutes === 1 ? 'minute' : 'minutes'} observed`;
    context.textContent = active.minutes === 0 ? 'No camera observations in this window. That does not tell us whether the dog was here.' :
      active.minutes < 4 ? 'Very little of this hour was watched. No sightings here is not evidence that the dog was absent.' :
      active.events.length === 0 ? 'No sightings in the observed minutes. The rest of the hour is still unknown.' :
      'These sightings belong to the minutes the camera watched, not to the whole hour.';
    const daylight = Math.max(0, Math.sin((hour - 5.5) / 14 * Math.PI));
    scene.querySelector('.room-day__night').style.opacity = state.appearance ? String((1 - daylight) * .44) : '0';
    const sun = scene.querySelector('.room-day__sun');
    sun.style.opacity = state.appearance ? String(daylight * .8) : '0';
    sun.setAttribute('transform', `translate(${(hour - 12) * 7} 0)`);
    dots.replaceChildren();
    PATCHES.forEach(patch => {
      const q = sample(patch.id, hour), views = zoneNodes.get(patch.id);
      views.coverage.style.opacity = state.coverage ? String(.04 + q.minutes / 60 * .65) : '0';
      views.hatch.style.opacity = state.coverage && q.minutes < 4 ? '.75' : '0';
      views.g.dataset.selected = String(state.patch === patch.id);
      patchButtons.get(patch.id).setAttribute('aria-pressed', String(state.patch === patch.id));
      if (state.sightings) q.events.forEach(event => {
        const cx = patch.x + 38 + (event.index * 37 % Math.max(40, patch.w - 72));
        const cy = patch.y + 35 + (event.index * 23 % Math.max(40, patch.h - 68));
        dots.append(svgNode('circle', {cx, cy, r: 11, class: 'room-day__dot-halo'}),
          svgNode('circle', {cx, cy, r: 5, class: 'room-day__dot'}));
      });
    });
    presetNodes.forEach(p => p.button.setAttribute('aria-pressed', String(Math.abs(p.hour - hour) < .01)));
    play.textContent = state.playing ? 'Pause day' : 'Play day'; play.setAttribute('aria-pressed', String(state.playing));
    play.disabled = motion.matches;
    motionNote.textContent = motion.matches ? 'Reduced motion is on. Use the slider or time buttons to explore.' :
      state.coverage ? 'Hatching marks places the camera scarcely watched.' : 'Turn on camera coverage to see the gaps in observation.';
    if (announce) live.textContent = `${clock(hour)}. ${active.patch.name}: ${counts.textContent}. ${context.textContent}`;
  }
  function stop() {
    state.playing = false; previous = 0;
    if (request !== null) cancelAnimationFrame(request);
    request = null; update();
  }
  function tick(now) {
    if (!state.playing) return;
    const box = scene.getBoundingClientRect();
    if (motion.matches || document.hidden || !state.visible || box.bottom <= 0 || box.top >= window.innerHeight) { stop(); return; }
    if (previous) state.hour = Math.min(24, state.hour + (now - previous) / 1000 * .4);
    previous = now;
    if (now - lastPaint >= 80) { update(); lastPaint = now; }
    if (state.hour >= 24) { stop(); update(true); return; }
    request = requestAnimationFrame(tick);
  }
  play.addEventListener('click', () => {
    if (state.playing) { stop(); return; }
    if (motion.matches || document.hidden) return;
    if (state.hour >= 24) state.hour = 0;
    state.playing = true; previous = 0; update(); request = requestAnimationFrame(tick);
  });
  range.addEventListener('input', () => { const hour = clampHour(range.value); stop(); state.hour = hour; update(true); });
  scene.addEventListener('click', event => {
    const patch = event.target.closest('[data-day-patch]');
    if (patch) { state.patch = patch.dataset.dayPatch; update(true); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  const changedMotion = () => { if (motion.matches) stop(); update(); };
  if (motion.addEventListener) motion.addEventListener('change', changedMotion); else motion.addListener(changedMotion);
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      state.visible = entries[0].isIntersecting;
      if (!state.visible) stop();
    }, {threshold: 0});
    observer.observe(scene);
  } else state.visible = true;
  root.dataset.dayReady = 'true'; update();
})();
