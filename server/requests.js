/* Request workflow and evidence stay separate. No dependency on the camera player. */
(function (global) {
  'use strict';
  const LANES = [
    ['not_started', 'Backlog'], ['in_progress', 'In progress'],
    ['monitoring', 'Monitoring'], ['needs_review', 'Needs review'], ['blocked', 'Blocked'], ['done', 'Done'],
  ];
  const labels = Object.fromEntries(LANES);
  const laneDescriptions = {
    not_started: 'Work or investigation still needed.', in_progress: 'Being worked on now.',
    monitoring: 'Measured goals and behavior to watch. No decision needed now.',
    needs_review: 'Evidence is ready for your review or decision.',
    blocked: 'Waiting on hardware or another dependency.', done: 'Every requested part is verified.',
  };
  const needsReason = state => state === 'blocked' || state === 'needs_review';
  const goalTitles = new Map([['search_success', 'Find locates your pet'], ['find_time', 'Find locates your pet quickly']]);
  const goalTitle = row => goalTitles.get(row.goal_id) || row.title || row.goal_id;
  const isGoal = row => row?.kind === 'goal' && typeof row.goal_id === 'string' && row.id === 'goal:' + row.goal_id;
  const compareRows = (a, b) => Number(isGoal(a)) - Number(isGoal(b)) || Number(b.ts) - Number(a.ts) || b.id.localeCompare(a.id);
  const normalizeGoals = goals => {
    const found = new Map();
    if (!Array.isArray(goals)) throw new Error('The goal monitor response is incomplete.');
    for (const row of goals) {
      if (!row || row.kind !== 'goal' || typeof row.goal_id !== 'string' || !/^[A-Za-z0-9_.-]+$/.test(row.goal_id)) throw new Error('A goal monitor has an invalid identity.');
      const id = 'goal:' + row.goal_id;
      if (found.has(id)) throw new Error('A goal monitor was returned more than once.');
      found.set(id, { ...row, id, kind: 'goal', movable: false, ts: row.sample?.ts || null, status: { state: 'monitoring' } });
    }
    return [...found.values()];
  };
  const goalHealth = (row, now = Date.now() / 1000) => {
    if (row.transport_error) return { state: 'unavailable', label: 'Refresh unavailable', reason: 'The last saved measurement is shown; current goal health could not be checked.' };
    const health = row.health;
    if (!health || !['on_target', 'off_target', 'stale', 'unmeasured', 'invalid'].includes(health.state))
      return { state: 'invalid', label: 'Unknown', reason: 'No trustworthy goal status is available.' };
    if (['on_target', 'off_target'].includes(health.state)) {
      if (typeof row.sample?.value !== 'number' || !Number.isFinite(row.sample.value) || !Number.isFinite(health.expires_at))
        return { state: 'invalid', label: 'Unknown', reason: 'The measurement is missing or invalid.' };
      if (Number.isFinite(health.expires_at) && now > health.expires_at)
        return { state: 'stale', label: 'Stale', reason: 'This measurement is too old to describe current health.' };
    }
    return health;
  };
  const goalValue = row => typeof row.sample?.value === 'number' && Number.isFinite(row.sample.value) ? String(row.sample.value) : '—';
  const startupSeconds = value => typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value.toFixed(3).replace(/\.?0+$/, '') + ' s' : '—';
  const startupStats = row => {
    const raw = row.goal_id === 'video_start' ? row.sample?.startup : null;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
    const count = value => Number.isInteger(value) && value >= 0 ? value : null;
    const min = Math.max(5, count(raw.min_samples) || 5), pageOpen = raw.from === 'open';
    const n = pageOpen ? count(raw.n) : null, p50 = pageOpen ? number(raw.p50) : null, p90 = pageOpen ? number(raw.p90) : null;
    return { n, p50, p90, min_samples: min, ready: raw.ready === true && n !== null && n >= min && p50 !== null && p90 !== null,
      window_days: count(raw.window_days), timeouts: count(raw.timeouts), reconnects: count(raw.reconnects),
      connect_n: count(raw.connect_n), connect_p50: number(raw.connect_p50), connect_p90: number(raw.connect_p90) };
  };
  const startupReading = row => {
    const stats = startupStats(row);
    if (!stats) return { value: 'Median —', detail: 'Page-open median and P90 unavailable' };
    const sampleCount = stats.n === null ? 'sample count unavailable' : `${stats.n} page openings`;
    return { value: 'Median ' + (stats.ready ? startupSeconds(stats.p50) : '—'),
      detail: `P90 ${stats.ready ? startupSeconds(stats.p90) : '—'} · ${sampleCount}${stats.ready ? '' : ` · need ${stats.min_samples} valid samples`}` };
  };
  const measurementAge = (row, now = Date.now() / 1000) => {
    const sourceKnown = typeof row.sample?.observed_at === 'number' && Number.isFinite(row.sample.observed_at);
    const measured = sourceKnown ? row.sample.observed_at : row.sample?.ts;
    const label = sourceKnown ? goalValue(row) === '—' ? 'Checked' : 'Measured' : 'Collected';
    if (typeof measured !== 'number' || !Number.isFinite(measured)) return 'Measurement time unavailable';
    const age = now - measured;
    if (age < -60) return 'Measurement time is ahead of this device';
    if (age < 60) return `${label} just now`;
    if (age < 3600) return `${label} ${Math.floor(age / 60)}m ago`;
    if (age < 86400) return `${label} ${Math.floor(age / 3600)}h ago`;
    return `${label} ${Math.floor(age / 86400)}d ago`;
  };
  const trendSegments = (points, maxGap = 90 * 60) => {
    const data = Array.isArray(points) ? points.slice(-200) : [];
    const valid = data.filter(p => Array.isArray(p) && Number.isFinite(p[0]) && typeof p[1] === 'number' && Number.isFinite(p[1]));
    if (valid.length < 2) return [];
    const xs = valid.map(p => p[0]), ys = valid.map(p => p[1]), xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const segments = []; let path = '', previous = null;
    for (const p of data) {
      if (!Array.isArray(p) || !Number.isFinite(p[0]) || typeof p[1] !== 'number' || !Number.isFinite(p[1])) { if (path) segments.push(path); path = ''; previous = null; continue; }
      if (previous != null && (p[0] < previous || p[0] - previous > maxGap)) { if (path) segments.push(path); path = ''; }
      const x = 2 + (p[0] - xmin) / (xmax - xmin || 1) * 296, y = 38 - (p[1] - ymin) / (ymax - ymin || 1) * 34;
      path += `${path ? ' L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
      previous = p[0];
    }
    if (path) segments.push(path);
    return segments;
  };
  const laneOf = row => {
    if (isGoal(row)) return 'monitoring';
    const state = row.status?.state;
    if (state === 'needs_hardware') return 'blocked';
    return typeof state === 'string' && Object.prototype.hasOwnProperty.call(labels, state) ? state : 'not_started';
  };
  const hasEvidence = value => typeof value === 'string' ? !!value.trim()
    : value && typeof value === 'object' ? Object.values(value).some(hasEvidence) : false;
  const components = row => Array.isArray(row.resolution?.components)
    ? row.resolution.components.filter(p => p && typeof p === 'object' && !Array.isArray(p)) : [];
  const reviewReady = part => !!part?.review && typeof part.review.question === 'string' && !!part.review.question.trim()
    && (typeof part.review.evidence === 'string' || Array.isArray(part.review.evidence)) && hasEvidence(part.review.evidence);
  const componentState = part => {
    if (!part || typeof part !== 'object' || Array.isArray(part) || typeof part.state !== 'string') return 'not_started';
    const state = part.state;
    if (state !== 'needs_hardware' && !Object.prototype.hasOwnProperty.call(labels, state)) return 'not_started';
    if (state === 'done' && !hasEvidence(part.evidence) || state === 'needs_review' && !reviewReady(part)) return 'not_started';
    return state;
  };
  const componentLabel = part => componentState(part) === 'needs_hardware' ? 'Needs hardware' : labels[componentState(part)];
  const readyReplyReview = row => row.resolution == null && row.reply?.status === 'answered'
    && ['live', 'no_change', 'planned'].includes(row.reply.fix?.status)
    && reviewReady({ review: row.reply.fix?.review }) ? row.reply.fix.review : null;
  const reviewQuestion = row => {
    if (laneOf(row) !== 'needs_review') return null;
    if (row.workflow?.state === 'needs_review' && typeof row.workflow.reason === 'string' && row.workflow.reason.trim()) return row.workflow.reason;
    const part = components(row).find(p => componentState(p) === 'needs_review');
    return part?.review.question || readyReplyReview(row)?.question || null;
  };
  const nonempty = value => typeof value === 'string' && !!value.trim();
  const actionPlans = row => {
    if (isGoal(row) || laneOf(row) === 'done') return [];
    return components(row).filter(part => componentState(part) !== 'done').flatMap(part => {
      const action = part.next_action;
      if (!action || !['owner', 'agent'].includes(action.assignee)
          || !['task', 'measure', 'done_when'].every(key => nonempty(action[key]))) return [];
      const steps = Array.isArray(action.steps) ? action.steps.filter(nonempty).map(step => step.trim()) : [];
      if (action.assignee === 'owner' && (!steps.length || !nonempty(action.on_failure))) return [];
      return [{ ...action, task: action.task.trim(), measure: action.measure.trim(), done_when: action.done_when.trim(),
        steps, record_template: nonempty(action.record_template) && action.record_template.length <= 2000 ? action.record_template.trim() : null,
        component: part.title || part.id }];
    });
  };
  const actionSummary = row => {
    const plans = actionPlans(row), owner = plans.find(action => action.assignee === 'owner');
    if (owner) return owner.task;
    const question = reviewQuestion(row);
    if (question) return question;
    const agent = plans.find(action => action.assignee === 'agent');
    return agent ? 'Agent: ' + agent.task : null;
  };
  const reportFrame = row => typeof row.id === 'string' && /^\d{8}-\d{6}$/.test(row.id)
    && row.evidence?.frame === '/api/reports/frame?id=' + row.id ? row.evidence.frame : null;
  const progress = row => {
    const parts = components(row);
    return { done: parts.filter(p => p.state === 'done' && hasEvidence(p.evidence)).length, total: parts.length };
  };
  const matches = (row, query) => !query.trim() ||
    [row.note, row.by, isGoal(row) ? goalTitle(row) : row.title, row.metric, row.target, row.source, row.reply?.text, row.status?.reason,
      isGoal(row) ? JSON.stringify([row.health, row.sample?.detail, row.work]) : '', JSON.stringify(row.resolution || {}), JSON.stringify(readyReplyReview(row) || {})]
      .join(' ').toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());

  class Store {
    constructor(api, changed = () => {}) {
      this.api = api; this.changed = changed; this.items = []; this.total = 0;
      this.loading = false; this.error = ''; this.epoch = 0; this.pending = null;
      this.goals = []; this.goalError = ''; this.goalsLoading = false;
      this.monitorTimeoutMs = 12000;
    }
    refresh() {
      if (this.pending) return this.pending;
      this.pending = Promise.all([this.fetchAll(), this.fetchGoals()]).finally(() => { this.pending = null; });
      return this.pending;
    }
    allItems() {
      return [...this.items.map(row => ({ ...row, kind: 'report' })), ...this.goals.map(row => ({ ...row, transport_error: this.goalError || null }))];
    }
    async fetchGoals() {
      this.goalsLoading = true; this.changed();
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      let timer;
      try {
        const timeout = new Promise((_, reject) => { timer = setTimeout(() => { controller?.abort(); reject(new Error('The goal monitor request timed out.')); }, this.monitorTimeoutMs); });
        const response = await Promise.race([this.api('/api/monitoring', undefined, { signal: controller?.signal }), timeout]);
        if (!response || response.available === false) throw new Error(response?.error || 'The monitor data is unavailable.');
        this.goals = normalizeGoals(response.goals); this.goalError = '';
      } catch (error) { this.goalError = `Couldn't refresh goal monitors: ${error.message}`; }
      finally { clearTimeout(timer); this.goalsLoading = false; this.changed(); }
    }
    async fetchAll() {
      this.loading = true; this.error = ''; this.changed();
      const epoch = this.epoch, found = new Map(), cursors = new Set();
      let before = null;
      try {
        do {
          const page = await this.api('/api/reports?page=1&limit=100' + (before ? '&before=' + encodeURIComponent(before) : ''));
          const incoming = Array.isArray(page) ? page : page.reports || [];
          incoming.forEach(row => found.set(row.id, row));
          this.total = Array.isArray(page) ? page.length : page.total;
          before = Array.isArray(page) ? null : page.next_before;
          if (before && cursors.has(before)) throw new Error('The history cursor stopped advancing. Please retry.');
          if (before) cursors.add(before);
        } while (before);
        if (epoch === this.epoch) this.items = [...found.values()];
      } catch (error) {
        if (!this.items.length && epoch === this.epoch) this.items = [...found.values()];
        this.error = `Couldn't load all requests: ${error.message}`;
      } finally { this.loading = false; this.changed(); }
    }
    async move(id, state, reason = '', displayedRevision) {
      if (typeof id === 'string' && id.startsWith('goal:')) throw new Error('Automatic goals stay in Monitoring. Their linked work is shown separately.');
      const row = this.items.find(r => r.id === id);
      if (!row) throw new Error('This request is no longer on the board. Refresh and try again.');
      if (needsReason(state) && !reason.trim()) throw new Error(state === 'needs_review'
        ? 'Add what you can review or decide now.' : 'Add what is blocking this request.');
      const updated = await this.api('/api/reports/move', { id, state, revision: displayedRevision ?? row.revision, reason });
      this.epoch++;
      this.items = this.items.map(r => r.id === id ? updated : r);
      this.changed();
      return updated;
    }
    counts(query = '') {
      const counts = Object.fromEntries(LANES.map(([state]) => [state, 0]));
      this.allItems().filter(row => matches(row, query)).forEach(row => counts[laneOf(row)]++);
      return counts;
    }
  }

  function mount({ api, snapshot, onSent = () => {} }) {
    const $ = id => document.getElementById(id), root = $('report-sheet');
    const board = $('report-list'), detail = $('report-detail'), tabs = $('report-tabs');
    let active = false, selectedLane = 'needs_review', selectedId = null, query = '', timer = null, openEpoch = 0, goalNavigation = 0;
    try {
      const saved = localStorage.getItem('openpetcam.requests.lane');
      if (LANES.some(([state]) => state === saved)) selectedLane = saved;
    } catch (_) { /* Storage is optional in private browsing. */ }
    let rendered = '', detailRevision = '', detailDirty = false, saving = false, message = '', firstLoad = true;
    const store = new Store(api, render);
    const make = (tag, cls, text) => {
      const el = document.createElement(tag); if (cls) el.className = cls;
      if (text !== undefined) el.textContent = text;
      return el;
    };
    const when = ts => ts != null && Number.isFinite(Number(ts)) ? new Date(Number(ts) * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
    }) : 'Date unavailable';
    const say = text => { message = text; $('report-status').textContent = text; };
    const rows = () => store.allItems().filter(row => matches(row, query)).sort(compareRows);
    const revisionOf = row => isGoal(row) ? JSON.stringify([row, goalHealth(row)]) : row.revision;

    function chooseLane(state, reveal = false) {
      selectedLane = state;
      try { localStorage.setItem('openpetcam.requests.lane', state); } catch (_) { /* Optional preference. */ }
      tabs.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lane === state)));
      board.querySelectorAll('.request-lane').forEach(lane => { lane.dataset.selected = String(lane.dataset.lane === state); });
      if (!tabs.classList.contains('hidden')) tabs.querySelector(`[data-lane="${state}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      if (reveal && !board.classList.contains('hidden')) board.querySelector(`.request-lane[data-lane="${state}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
    function card(row) {
      const item = make('li'), button = make('button', 'request-card');
      button.type = 'button'; button.dataset.id = row.id; button.draggable = !isGoal(row);
      if (isGoal(row)) {
        button.classList.add('request-goal-card');
        button.append(make('span', 'request-card-kind', 'Automatic goal'));
        button.append(make('span', 'request-card-title', goalTitle(row)));
        const health = goalHealth(row), reading = make('span', 'request-goal-reading');
        const startup = row.goal_id === 'video_start' ? startupReading(row) : null;
        reading.append(make('strong', '', startup ? startup.value : goalValue(row)), make('span', 'request-goal-health', health.label));
        reading.dataset.health = health.state; button.append(reading);
        if (startup) button.append(make('span', 'request-card-next', startup.detail));
        button.append(make('span', 'request-card-next', 'Target ' + (row.target || 'not configured')));
        button.append(make('span', 'request-card-meta', measurementAge(row)));
        button.onclick = () => openCard(row.id); item.append(button); return item;
      }
      if (laneOf(row) === 'monitoring') button.append(make('span', 'request-card-kind', 'Owner observation'));
      button.append(make('span', 'request-card-title', row.note || 'Request without a note'));
      const p = progress(row), remaining = components(row).find(c => componentState(c) !== 'done'), next = actionSummary(row);
      if (next) button.append(make('span', 'request-card-next', next));
      else if (remaining) button.append(make('span', 'request-card-next', remaining.title || remaining.id));
      else if (row.status?.reason && laneOf(row) !== 'done') button.append(make('span', 'request-card-next', row.status.reason));
      const meta = make('span', 'request-card-meta');
      meta.append(make('span', '', when(row.ts)));
      if (p.total) meta.append(make('span', 'request-progress', `${p.done}/${p.total} verified`));
      button.append(meta); button.onclick = () => openCard(row.id);
      button.addEventListener('dragstart', event => {
        event.dataTransfer.setData('text/plain', row.id); event.dataTransfer.effectAllowed = 'move';
      });
      item.append(button); return item;
    }
    function render() {
      const visible = rows(), counts = store.counts(query);
      if (!store.loading && !store.goalsLoading && store.allItems().length && firstLoad) {
        firstLoad = false;
        if (!counts[selectedLane]) selectedLane = LANES.find(([state]) => counts[state])?.[0] || selectedLane;
      }
      $('report-history').textContent = `${store.loading ? 'Loading requests…' : `${store.items.length} of ${store.total} requests`} · ${store.goals.length} goal monitors${store.goalsLoading ? ' (checking…)' : ''}${query ? ` · ${visible.length} matching` : ''}`;
      $('report-status').textContent = [store.error, store.goalError, message].filter(Boolean).join(' ');
      $('report-retry').classList.toggle('hidden', !store.error && !store.goalError);
      board.setAttribute('aria-busy', String(store.loading || store.goalsLoading));
      const signature = JSON.stringify([visible, counts, visible.filter(isGoal).map(row => [goalHealth(row), measurementAge(row)]), store.goalError]);
      if (signature !== rendered) {
        rendered = signature;
        const positions = Object.fromEntries([...board.querySelectorAll('.request-cards')].map(el => [el.dataset.lane, el.scrollTop]));
        board.replaceChildren(); tabs.replaceChildren();
        for (const [state, label] of LANES) {
          const tab = make('button', 'request-tab', `${label} ${counts[state]}`);
          tab.type = 'button'; tab.dataset.lane = state; tab.title = laneDescriptions[state]; tab.onclick = () => chooseLane(state, true); tabs.append(tab);
          const lane = make('section', 'request-lane'); lane.dataset.lane = state;
          const heading = make('h4', '', label); heading.append(make('span', 'request-count', counts[state]));
          const list = make('ul', 'request-cards'); list.dataset.lane = state;
          visible.filter(row => laneOf(row) === state).forEach(row => list.append(card(row)));
          if (state === 'monitoring' && store.goalError) list.append(make('li', 'request-empty', 'Automatic goal refresh is unavailable. Existing request observations remain visible.'));
          if (!list.children.length) list.append(make('li', 'request-empty', store.loading ? 'Loading…' : query ? 'No matching requests' : 'No requests'));
          lane.append(heading, make('p', 'request-lane-description', laneDescriptions[state]), list); board.append(lane); list.scrollTop = positions[state] || 0;
          lane.addEventListener('dragover', event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; lane.classList.add('drop-target'); });
          lane.addEventListener('dragleave', () => lane.classList.remove('drop-target'));
          lane.addEventListener('drop', event => {
            event.preventDefault(); lane.classList.remove('drop-target');
            const id = event.dataTransfer.getData('text/plain');
            if (!store.items.some(row => row.id === id)) return;
            if (needsReason(state)) { openCard(id); $('request-state').value = state; showMoveReason(); $('request-reason').focus(); }
            else move(id, state);
          });
        }
        chooseLane(selectedLane);
      }
      const row = store.allItems().find(r => r.id === selectedId);
      if (row && revisionOf(row) !== detailRevision && !saving && !detailDirty) renderDetail(row);
    }
    function evidence(parent, value, key, label = 'Evidence and limitations') {
      if (!hasEvidence(value)) return;
      const fold = make('details', 'request-evidence'), summary = make('summary', '', label);
      fold.dataset.fold = key;
      const list = make('ul');
      const values = Array.isArray(value) ? value : [value];
      values.forEach(v => list.append(make('li', '', typeof v === 'string' ? v : JSON.stringify(v))));
      fold.append(summary, list); parent.append(fold);
    }
    function renderActionPlan(parent, action) {
      parent.append(make('p', 'request-action-task', action.task));
      if (action.steps.length) {
        parent.append(make('h6', '', action.assignee === 'owner' ? 'How to check' : 'Next steps'));
        const list = make('ol'); action.steps.forEach(step => list.append(make('li', '', step))); parent.append(list);
      }
      const facts = make('dl', 'request-action-facts');
      facts.append(make('dt', '', 'Measure'), make('dd', '', action.measure),
        make('dt', '', action.assignee === 'owner' ? 'Check complete when' : 'Ready for review when'), make('dd', '', action.done_when));
      if (nonempty(action.on_failure)) facts.append(make('dt', '', action.assignee === 'owner' ? 'Report when' : 'If the test fails'), make('dd', '', action.on_failure));
      parent.append(facts);
      if (action.assignee === 'owner' && action.record_template) {
        const record = make('button', 'btn request-record-result', 'Record result'); record.type = 'button';
        record.onclick = () => {
          $('report-new').onclick();
          const field = $('report-note');
          if (!field.value.trim()) field.value = action.record_template;
          else say('Your existing draft is preserved. Add the check result before sending.');
          field.focus();
        };
        parent.append(record);
      }
    }
    function renderNextActions(row) {
      const plans = actionPlans(row), owner = plans.filter(action => action.assignee === 'owner'), agents = plans.filter(action => action.assignee === 'agent');
      const section = make('section', 'request-next-action');
      section.append(make('h5', '', 'Your next step'));
      if (owner.length) owner.forEach(action => renderActionPlan(section, action));
      else if (reviewQuestion(row)) section.append(make('p', 'request-action-task', reviewQuestion(row)), make('p', '', 'Use the review evidence below to answer this question.'));
      else {
        section.append(make('p', 'request-action-task', 'No action needed from you now.'));
        if (laneOf(row) === 'monitoring') section.append(make('p', '', 'The agent still needs to define a concrete check.'));
      }
      detail.append(section);
      if (agents.length) {
        const work = make('section', 'request-agent-action'); work.append(make('h5', '', "Agent’s next step"));
        agents.forEach(action => renderActionPlan(work, action)); detail.append(work);
      }
    }
    function renderSavedFrame(row) {
      const source = reportFrame(row); if (!source) return;
      const figure = make('figure', 'request-saved-frame'), image = make('img');
      const caption = make('figcaption', '', 'Saved report frame. The phone status and this frame may be from different moments.');
      image.src = source; image.alt = 'Saved camera frame from this report'; image.loading = 'lazy'; image.decoding = 'async';
      image.onerror = () => { image.hidden = true; caption.textContent = 'The saved frame is unavailable. The request and its recorded status remain below.'; };
      figure.append(image, caption);
      if (nonempty(row.shown)) figure.append(make('p', '', 'Status recorded on the phone: ' + row.shown));
      detail.append(figure);
    }
    function renderDetail(row) {
      const same = detail.dataset.id === row.id, oldScroll = same ? detail.scrollTop : 0;
      const openFolds = new Set(same ? [...detail.querySelectorAll('details[open]')].map(el => el.dataset.fold) : []);
      const oldFocus = same && detail.contains(document.activeElement) ? document.activeElement : null;
      const focusId = oldFocus?.id, focusFold = oldFocus?.closest('details')?.dataset.fold;
      detailRevision = revisionOf(row); detailDirty = false; detail.replaceChildren();
      detail.dataset.id = row.id;
      const back = make('button', 'linkbtn request-back', '← Board'); back.type = 'button'; back.onclick = backToBoard;
      const title = make('h4', '', isGoal(row) ? goalTitle(row) : row.note || 'Request without a note'); title.tabIndex = -1; title.id = 'request-detail-title';
      if (isGoal(row)) {
        detail.append(back, title, make('p', 'request-detail-meta', 'Automatic goal · Monitoring'));
        renderGoalDetail(row);
        restoreDetail(openFolds, focusId, focusFold, oldScroll);
        return;
      }
      detail.append(back, title, make('p', 'request-detail-meta', `${when(row.ts)}${row.by ? ' · ' + row.by : ''}`));
      renderNextActions(row);
      const goalIds = [...new Set(components(row).map(part => part.goal_id).filter(id => typeof id === 'string' && /^[a-zA-Z0-9_.-]{1,100}$/.test(id)))];
      for (const goalId of goalIds) {
        const link = make('button', 'btn', goalIds.length === 1 ? 'View measured goal' : 'View measured goal: ' + goalId);
        link.type = 'button'; link.dataset.goal = goalId;
        link.onclick = () => openMeasuredGoal(goalId, row.id); detail.append(link);
      }
      renderSavedFrame(row);
      detail.append(make('p', 'request-status-reason', row.status?.reason || 'No verified resolution is recorded yet.'));
      if (laneOf(row) === 'needs_review' && row.workflow?.reason) {
        const review = make('section', 'request-review');
        review.append(make('h5', '', 'Ready for your review'), make('p', '', row.workflow.reason)); detail.append(review);
      }
      const replyReview = readyReplyReview(row);
      if (replyReview) {
        const review = make('section', 'request-review');
        review.append(make('h5', '', 'Ready for your review'), make('p', '', replyReview.question));
        evidence(review, replyReview.evidence, 'reply-review', 'Review evidence'); detail.append(review);
      }
      const controls = make('div', 'request-move');
      const label = make('label', '', 'Move to'); label.htmlFor = 'request-state';
      const select = make('select', 'field'); select.id = 'request-state';
      LANES.forEach(([state, text]) => { const option = make('option', '', text); option.value = state; select.append(option); });
      select.value = laneOf(row); select.onchange = () => { detailDirty = true; showMoveReason(); };
      const save = make('button', 'btn', 'Move'); save.type = 'button'; save.id = 'request-move-save';
      save.onclick = () => move(row.id, select.value, $('request-reason').value);
      controls.append(label, select, save); detail.append(controls);
      const reason = make('div', 'request-block-reason hidden'); reason.id = 'request-block-reason';
      const reasonLabel = make('label'); reasonLabel.id = 'request-reason-label'; reasonLabel.htmlFor = 'request-reason';
      const reasonInput = make('textarea', 'field'); reasonInput.id = 'request-reason'; reasonInput.maxLength = 1000; reasonInput.rows = 2;
      reasonInput.value = row.workflow?.reason || ''; reason.append(reasonLabel, reasonInput); detail.append(reason);
      reasonInput.oninput = () => { detailDirty = true; showMoveReason(); };
      showMoveReason();
      const parts = components(row);
      if (parts.length) {
        detail.append(make('h5', '', 'Requested work'));
        const list = make('ul', 'request-checklist');
        parts.forEach(part => {
          const item = make('li');
          item.append(make('strong', '', part.title || part.id), make('span', 'request-part-state', componentLabel(part)));
          if (part.reason) item.append(make('p', '', part.reason));
          if (part.state === 'needs_review' && reviewReady(part)) {
            const review = make('section', 'request-review');
            review.append(make('h5', '', 'Ready for your review'), make('p', '', part.review.question));
            evidence(review, part.review.evidence, 'review-' + String(part.id || part.title), 'Review evidence'); item.append(review);
          }
          evidence(item, part.evidence, String(part.id || part.title)); list.append(item);
        });
        detail.append(list);
      }
      if (row.reply?.text) {
        const discussion = make('details', 'request-discussion'), summary = make('summary', '', 'Discussion');
        discussion.dataset.fold = 'discussion';
        discussion.append(summary, make('div', '', row.reply.text)); detail.append(discussion);
      }
      restoreDetail(openFolds, focusId, focusFold, oldScroll);
    }
    function restoreDetail(openFolds, focusId, focusFold, oldScroll) {
      detail.querySelectorAll('details').forEach(el => { if (openFolds.has(el.dataset.fold)) el.open = true; });
      if (focusId && $(focusId)) $(focusId).focus({ preventScroll: true });
      else if (focusFold) [...detail.querySelectorAll('details')].find(el => el.dataset.fold === focusFold)?.querySelector('summary')?.focus({ preventScroll: true });
      detail.scrollTop = oldScroll;
    }
    function renderGoalDetail(row) {
      const health = goalHealth(row), status = make('p', 'request-goal-status', health.label);
      status.dataset.health = health.state; detail.append(status);
      if (health.reason) detail.append(make('p', 'request-status-reason', health.reason));
      const readings = make('dl', 'request-goal-facts');
      const fact = (label, value) => { readings.append(make('dt', '', label), make('dd', '', value)); };
      if (row.goal_id === 'video_start') {
        const stats = startupStats(row);
        fact('Page-open median', stats?.ready ? startupSeconds(stats.p50) : 'Not assessed');
        fact('Page-open p90', stats?.ready ? startupSeconds(stats.p90) : 'Not assessed');
        fact('Page-open samples', stats?.n == null ? 'Unavailable' : String(stats.n));
        fact('Window', stats?.window_days ? `Last ${stats.window_days} days` : 'Unavailable');
        fact('Measurement', 'Page load to video playing for successful starts. Failed starts are outside these percentiles; foreground resumes are not measured as page opens.');
        fact('Startup timeout events', stats?.timeouts == null ? 'Unavailable' : String(stats.timeouts));
        fact('Reconnect events', stats?.reconnects == null ? 'Unavailable' : String(stats.reconnects));
        if (stats?.connect_n != null) {
          fact('Connection-only samples', String(stats.connect_n));
          fact('Connection-only median', startupSeconds(stats.connect_p50));
          fact('Connection-only p90', startupSeconds(stats.connect_p90));
          fact('Connection diagnostics', 'Connection to video playing; these timings do not satisfy the page-open target.');
        }
        if (!stats && goalValue(row) !== '—') fact('Legacy recorded value', goalValue(row) + ' (measurement basis unverified)');
      } else fact('Latest value', goalValue(row));
      fact('Target', row.target || 'Not configured');
      fact('Measured at', row.sample?.observed_at != null ? when(row.sample.observed_at) : 'Source measurement time unavailable');
      if (row.sample?.ts != null) fact('Collected at', when(row.sample.ts));
      if (row.sample?.target != null && String(row.sample.target) !== String(row.target)) fact('Sample target', String(row.sample.target));
      if (row.metric) fact('Metric', row.metric); if (row.source) fact('Source', row.source);
      detail.append(readings);
      if (row.note) detail.append(make('p', 'request-status-reason', row.note));
      if (row.sample?.detail != null) evidence(detail, row.sample.detail, 'goal-detail', 'Measurement details and limitations');
      const segments = trendSegments(row.series);
      if (segments.length) {
        detail.append(make('h5', '', 'Recorded values over time'));
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 300 42'); svg.setAttribute('class', 'request-goal-trend');
        svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Recorded values. Recorded missing values and gaps over 90 minutes break the line. Skipped collections are not shown.');
        for (const d of segments) { const path = document.createElementNS('http://www.w3.org/2000/svg', 'path'); path.setAttribute('d', d); svg.append(path); }
        detail.append(svg, make('p', 'request-detail-meta', 'Recorded values only; skipped collections are not shown. Gaps over 90 minutes break the line.'));
      }
      detail.append(make('h5', '', 'Linked work'));
      const work = Array.isArray(row.work) ? row.work.filter(thread => thread && typeof thread === 'object' && !Array.isArray(thread)) : [];
      if (!work.length) detail.append(make('p', 'request-status-reason', 'No linked work is recorded yet.'));
      for (const thread of work) {
        const section = make('section', 'request-goal-work');
        section.append(make('strong', '', thread.title || thread.id || 'Agent work'), make('p', '', String(thread.status || 'Status unavailable')));
        if (thread.summary) section.append(make('p', '', thread.summary));
        if (thread.updated != null) section.append(make('p', 'request-detail-meta', 'Updated ' + when(thread.updated)));
        if (typeof thread.id === 'string') { const link = make('a', 'linkbtn', 'View agent activity'); link.href = '/agent?thread=' + encodeURIComponent(thread.id); section.append(link); }
        detail.append(section);
      }
      detail.append(make('p', 'request-goal-readonly', 'This goal is checked periodically. Linked work tracks any changes; the measurement and target stay here.'));
    }
    function showMoveReason() {
      const state = $('request-state')?.value, required = needsReason(state);
      $('request-block-reason')?.classList.toggle('hidden', !required && state !== 'monitoring');
      const label = $('request-reason-label'), input = $('request-reason');
      if (label) label.textContent = state === 'needs_review' ? 'What should I review or decide?'
        : state === 'monitoring' ? 'What should I keep an eye on? (optional)' : 'What is blocking this?';
      if (input) input.required = required;
      if ($('request-move-save')) $('request-move-save').disabled = saving || (required && !input?.value.trim());
    }
    async function openMeasuredGoal(goalId, reportId) {
      if (!/^[a-zA-Z0-9_.-]{1,100}$/.test(goalId)) return;
      if (saving || detailDirty) { say('Save your request changes before opening this goal.'); return; }
      const navigation = ++goalNavigation, epoch = openEpoch;
      say('Checking goal measurements…');
      await store.fetchGoals();
      if (!active || epoch !== openEpoch || navigation !== goalNavigation || selectedId !== reportId || saving || detailDirty) return;
      const id = 'goal:' + goalId;
      if (store.goalError || !store.goals.some(row => row.id === id)) {
        say('Goal measurements are unavailable. Your request is still open.'); return;
      }
      say(''); chooseLane('monitoring'); openCard(id);
    }
    function openCard(id) {
      const row = store.allItems().find(r => r.id === id); if (!row) return;
      selectedId = id; selectedLane = laneOf(row); renderDetail(row);
      detail.classList.remove('hidden'); board.classList.add('hidden'); tabs.classList.add('hidden');
      $('report-search-label').classList.add('hidden'); $('request-detail-title').focus();
    }
    function backToBoard() {
      const id = selectedId; selectedId = null; detailRevision = ''; detailDirty = false;
      detail.classList.add('hidden'); board.classList.remove('hidden'); tabs.classList.remove('hidden');
      $('report-search-label').classList.remove('hidden'); chooseLane(selectedLane, true);
      board.querySelector(`[data-id="${id}"]`)?.focus();
    }
    async function move(id, state, reason = '') {
      if (saving) return;
      const displayedRevision = selectedId === id ? detailRevision : undefined;
      let failed = false, outcome = '';
      saving = true;
      for (const control of detail.querySelectorAll('select,textarea,button')) control.disabled = true;
      detail.setAttribute('aria-busy', 'true');
      try {
        const row = await store.move(id, state, reason, displayedRevision); selectedLane = laneOf(row);
        outcome = `Moved to ${labels[selectedLane]}.`;
      } catch (error) {
        failed = true; outcome = error.message;
        if (store.pending) await store.pending;
        await store.refresh();
      }
      finally {
        saving = false; detailRevision = ''; detailDirty = false; render();
        if (failed && selectedId === id && $('request-state')) {
          $('request-state').value = state; $('request-reason').value = reason; detailDirty = true; showMoveReason();
        }
        if (selectedId === id && !detail.classList.contains('hidden')) $('request-state')?.focus({ preventScroll: true });
        detail.setAttribute('aria-busy', 'false'); say(outcome);
      }
    }
    async function refresh() {
      clearTimeout(timer); await store.refresh();
      if (active) timer = setTimeout(refresh, 20000);
    }
    function closeComposer() {
      $('report-compose').classList.add('hidden'); $('report-dialog').classList.remove('composing'); $('report-new').focus();
    }
    function close() {
      active = false; openEpoch++; clearTimeout(timer); root.classList.add('hidden'); $('report-open').focus();
    }
    $('report-cancel').onclick = close;
    root.addEventListener('click', event => { if (event.target === root) close(); });
    root.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        if (!$('report-compose').classList.contains('hidden')) closeComposer();
        else if (selectedId) backToBoard(); else close();
      }
      if (event.key === 'Tab') {
        const focusable = [...root.querySelectorAll('button,input,textarea,select,summary,a[href],[tabindex="0"]')].filter(el => !el.disabled && el.getClientRects().length);
        const first = focusable[0], last = focusable.at(-1);
        if (!focusable.includes(document.activeElement) || (event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
          event.preventDefault(); (event.shiftKey ? last : first)?.focus();
        }
      }
    });
    $('report-search').addEventListener('input', event => {
      query = event.target.value;
      const counts = store.counts(query);
      if (!counts[selectedLane]) selectedLane = LANES.find(([state]) => counts[state])?.[0] || selectedLane;
      render();
    });
    $('report-retry').onclick = () => { say(''); refresh(); };
    $('report-new').onclick = () => {
      $('report-compose').classList.remove('hidden');
      $('report-dialog').classList.add('composing');
      const snap = snapshot(); $('report-compose').snapshot = snap;
      // Keep the picture being described; without it the report still goes out with the picture at sending.
      Promise.resolve().then(() => api('/api/report/hold', {})).catch(() => {});
      $('report-shown').textContent = `The app shows: ${snap.shown_status || ''}`;
      $('report-note').focus();
    };
    $('report-compose-cancel').onclick = closeComposer;
    $('report-send').onclick = async () => {
      const field = $('report-note'), note = field.value.trim(); if (!note) { field.focus(); return; }
      $('report-send').disabled = true; say('');
      try {
        const result = await api('/api/report', { note, client: $('report-compose').snapshot || snapshot() });
        field.value = ''; closeComposer(); onSent();
        if (store.pending) await store.pending;
        await store.refresh();
        if (store.items.some(row => row.id === result.id)) openCard(result.id);
        say('Request sent.');
      } catch (error) { say(`Couldn't send: ${error.message}`); }
      finally { $('report-send').disabled = false; }
    };
    return { store, close, async open(options = {}) {
      const epoch = ++openEpoch;
      active = true; root.classList.remove('hidden');
      $('report-compose').classList.add('hidden');
      $('report-dialog').classList.remove('composing');
      if (selectedId) backToBoard();
      $('report-dialog').focus();
      if (LANES.some(([state]) => state === options.lane)) { selectedLane = options.lane; firstLoad = false; chooseLane(selectedLane, true); }
      await refresh();
      if (!active || epoch !== openEpoch) return;
      if (typeof options.goal === 'string') {
        const id = options.goal.startsWith('goal:') ? options.goal : 'goal:' + options.goal;
        if (store.goals.some(row => row.id === id)) openCard(id); else say('That goal monitor is unavailable.');
      }
    } };
  }
  const exported = { LANES, laneDescriptions, Store, laneOf, progress, matches, components, hasEvidence, reviewReady, componentState, componentLabel, readyReplyReview, reviewQuestion, nonempty, actionPlans, actionSummary, reportFrame, needsReason, isGoal, compareRows, normalizeGoals, goalTitle, goalHealth, goalValue, startupSeconds, startupStats, startupReading, measurementAge, trendSegments, mount };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  global.OpenPetRequests = exported;
})(typeof window === 'undefined' ? globalThis : window);
