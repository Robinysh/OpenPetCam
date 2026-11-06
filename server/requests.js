/* Request workflow and evidence stay separate. No dependency on the camera player. */
(function (global) {
  'use strict';
  const LANES = [
    ['not_started', 'Backlog'], ['in_progress', 'In progress'],
    ['needs_review', 'Needs review'], ['blocked', 'Blocked'], ['done', 'Done'],
  ];
  const labels = Object.fromEntries(LANES);
  const laneOf = row => {
    const state = row.status?.state;
    if (state === 'needs_hardware') return 'blocked';
    return Object.prototype.hasOwnProperty.call(labels, state) ? state : row.reply ? 'needs_review' : 'not_started';
  };
  const hasEvidence = value => typeof value === 'string' ? !!value.trim()
    : value && typeof value === 'object' ? Object.values(value).some(hasEvidence) : false;
  const components = row => Array.isArray(row.resolution?.components)
    ? row.resolution.components.filter(p => p && typeof p === 'object' && !Array.isArray(p)) : [];
  const progress = row => {
    const parts = components(row);
    return { done: parts.filter(p => p.state === 'done' && hasEvidence(p.evidence)).length, total: parts.length };
  };
  const matches = (row, query) => !query.trim() ||
    [row.note, row.by, row.reply?.text, row.status?.reason, JSON.stringify(row.resolution || {})]
      .join(' ').toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());

  class Store {
    constructor(api, changed = () => {}) {
      this.api = api; this.changed = changed; this.items = []; this.total = 0;
      this.loading = false; this.error = ''; this.epoch = 0; this.pending = null;
    }
    refresh() {
      if (this.pending) return this.pending;
      this.pending = this.fetchAll().finally(() => { this.pending = null; });
      return this.pending;
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
      const row = this.items.find(r => r.id === id);
      if (!row) throw new Error('This request is no longer on the board. Refresh and try again.');
      const updated = await this.api('/api/reports/move', { id, state, revision: displayedRevision ?? row.revision, reason });
      this.epoch++;
      this.items = this.items.map(r => r.id === id ? updated : r);
      this.changed();
      return updated;
    }
    counts(query = '') {
      const counts = Object.fromEntries(LANES.map(([state]) => [state, 0]));
      this.items.filter(row => matches(row, query)).forEach(row => counts[laneOf(row)]++);
      return counts;
    }
  }

  function mount({ api, snapshot, onSent = () => {} }) {
    const $ = id => document.getElementById(id), root = $('report-sheet');
    const board = $('report-list'), detail = $('report-detail'), tabs = $('report-tabs');
    let active = false, selectedLane = 'needs_review', selectedId = null, query = '', timer = null;
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
    const rows = () => store.items.filter(row => matches(row, query))
      .sort((a, b) => Number(b.ts) - Number(a.ts) || b.id.localeCompare(a.id));

    function chooseLane(state) {
      selectedLane = state;
      try { localStorage.setItem('openpetcam.requests.lane', state); } catch (_) { /* Optional preference. */ }
      tabs.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lane === state)));
      board.querySelectorAll('.request-lane').forEach(lane => { lane.dataset.selected = String(lane.dataset.lane === state); });
      if (!tabs.classList.contains('hidden')) tabs.querySelector(`[data-lane="${state}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
    function card(row) {
      const item = make('li'), button = make('button', 'request-card');
      button.type = 'button'; button.dataset.id = row.id; button.draggable = true;
      button.append(make('span', 'request-card-title', row.note || 'Request without a note'));
      const p = progress(row), remaining = components(row).find(c => c.state !== 'done');
      if (remaining) button.append(make('span', 'request-card-next', remaining.title || remaining.id));
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
      if (!store.loading && store.items.length && firstLoad) {
        firstLoad = false;
        if (!counts[selectedLane]) selectedLane = LANES.find(([state]) => counts[state])?.[0] || selectedLane;
      }
      $('report-history').textContent = store.loading ? 'Loading requests…'
        : `${store.items.length} of ${store.total} requests${query ? ` · ${visible.length} matching` : ''}`;
      $('report-status').textContent = store.error || message;
      $('report-retry').classList.toggle('hidden', !store.error);
      board.setAttribute('aria-busy', String(store.loading));
      const signature = JSON.stringify([visible, counts]);
      if (signature !== rendered) {
        rendered = signature;
        const positions = Object.fromEntries([...board.querySelectorAll('.request-cards')].map(el => [el.dataset.lane, el.scrollTop]));
        board.replaceChildren(); tabs.replaceChildren();
        for (const [state, label] of LANES) {
          const tab = make('button', 'request-tab', `${label} ${counts[state]}`);
          tab.type = 'button'; tab.dataset.lane = state; tab.onclick = () => chooseLane(state); tabs.append(tab);
          const lane = make('section', 'request-lane'); lane.dataset.lane = state;
          const heading = make('h4', '', label); heading.append(make('span', 'request-count', counts[state]));
          const list = make('ul', 'request-cards'); list.dataset.lane = state;
          visible.filter(row => laneOf(row) === state).forEach(row => list.append(card(row)));
          if (!list.children.length) list.append(make('li', 'request-empty', store.loading ? 'Loading…' : query ? 'No matching requests' : 'No requests'));
          lane.append(heading, list); board.append(lane); list.scrollTop = positions[state] || 0;
          lane.addEventListener('dragover', event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; lane.classList.add('drop-target'); });
          lane.addEventListener('dragleave', () => lane.classList.remove('drop-target'));
          lane.addEventListener('drop', event => {
            event.preventDefault(); lane.classList.remove('drop-target');
            const id = event.dataTransfer.getData('text/plain');
            if (!store.items.some(row => row.id === id)) return;
            if (state === 'blocked') { openCard(id); $('request-state').value = state; showBlockReason(); $('request-reason').focus(); }
            else move(id, state);
          });
        }
        chooseLane(selectedLane);
      }
      const row = store.items.find(r => r.id === selectedId);
      if (row && row.revision !== detailRevision && !saving && !detailDirty) renderDetail(row);
    }
    function evidence(parent, value, key) {
      if (!hasEvidence(value)) return;
      const fold = make('details', 'request-evidence'), summary = make('summary', '', 'Evidence and limitations');
      fold.dataset.fold = key;
      const list = make('ul');
      const values = Array.isArray(value) ? value : [value];
      values.forEach(v => list.append(make('li', '', typeof v === 'string' ? v : JSON.stringify(v))));
      fold.append(summary, list); parent.append(fold);
    }
    function renderDetail(row) {
      const same = detail.dataset.id === row.id, oldScroll = same ? detail.scrollTop : 0;
      const openFolds = new Set(same ? [...detail.querySelectorAll('details[open]')].map(el => el.dataset.fold) : []);
      const oldFocus = same && detail.contains(document.activeElement) ? document.activeElement : null;
      const focusId = oldFocus?.id, focusFold = oldFocus?.closest('details')?.dataset.fold;
      detailRevision = row.revision; detailDirty = false; detail.replaceChildren();
      detail.dataset.id = row.id;
      const back = make('button', 'linkbtn request-back', '← Board'); back.type = 'button'; back.onclick = backToBoard;
      const title = make('h4', '', row.note || 'Request without a note'); title.tabIndex = -1; title.id = 'request-detail-title';
      detail.append(back, title, make('p', 'request-detail-meta', `${when(row.ts)}${row.by ? ' · ' + row.by : ''}`));
      detail.append(make('p', 'request-status-reason', row.status?.reason || 'No verified resolution is recorded yet.'));
      const controls = make('div', 'request-move');
      const label = make('label', '', 'Move to'); label.htmlFor = 'request-state';
      const select = make('select', 'field'); select.id = 'request-state';
      LANES.forEach(([state, text]) => { const option = make('option', '', text); option.value = state; select.append(option); });
      select.value = laneOf(row); select.onchange = () => { detailDirty = true; showBlockReason(); };
      const save = make('button', 'btn', 'Move'); save.type = 'button'; save.id = 'request-move-save';
      save.onclick = () => move(row.id, select.value, $('request-reason').value);
      controls.append(label, select, save); detail.append(controls);
      const reason = make('div', 'request-block-reason hidden'); reason.id = 'request-block-reason';
      const reasonLabel = make('label', '', 'What is blocking this?'); reasonLabel.htmlFor = 'request-reason';
      const reasonInput = make('textarea', 'field'); reasonInput.id = 'request-reason'; reasonInput.maxLength = 1000; reasonInput.rows = 2;
      reasonInput.value = row.workflow?.reason || ''; reason.append(reasonLabel, reasonInput); detail.append(reason);
      reasonInput.oninput = () => { detailDirty = true; };
      showBlockReason();
      const parts = components(row);
      if (parts.length) {
        detail.append(make('h5', '', 'Requested work'));
        const list = make('ul', 'request-checklist');
        parts.forEach(part => {
          const item = make('li');
          item.append(make('strong', '', part.title || part.id), make('span', 'request-part-state',
            part.state === 'needs_hardware' ? 'Needs hardware' : labels[part.state] || 'Needs review'));
          if (part.reason) item.append(make('p', '', part.reason));
          evidence(item, part.evidence, String(part.id || part.title)); list.append(item);
        });
        detail.append(list);
      }
      if (row.reply?.text) {
        const discussion = make('details', 'request-discussion'), summary = make('summary', '', 'Discussion');
        discussion.dataset.fold = 'discussion';
        discussion.append(summary, make('div', '', row.reply.text)); detail.append(discussion);
      }
      detail.querySelectorAll('details').forEach(el => { if (openFolds.has(el.dataset.fold)) el.open = true; });
      if (focusId && $(focusId)) $(focusId).focus({ preventScroll: true });
      else if (focusFold) [...detail.querySelectorAll('details')].find(el => el.dataset.fold === focusFold)?.querySelector('summary')?.focus({ preventScroll: true });
      detail.scrollTop = oldScroll;
    }
    function showBlockReason() {
      $('request-block-reason')?.classList.toggle('hidden', $('request-state')?.value !== 'blocked');
    }
    function openCard(id) {
      const row = store.items.find(r => r.id === id); if (!row) return;
      selectedId = id; selectedLane = laneOf(row); renderDetail(row);
      detail.classList.remove('hidden'); board.classList.add('hidden'); tabs.classList.add('hidden');
      $('report-search-label').classList.add('hidden'); $('request-detail-title').focus();
    }
    function backToBoard() {
      const id = selectedId; selectedId = null; detailRevision = ''; detailDirty = false;
      detail.classList.add('hidden'); board.classList.remove('hidden'); tabs.classList.remove('hidden');
      $('report-search-label').classList.remove('hidden'); chooseLane(selectedLane);
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
          $('request-state').value = state; $('request-reason').value = reason; detailDirty = true; showBlockReason();
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
      active = false; clearTimeout(timer); root.classList.add('hidden'); $('report-open').focus();
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
    return { store, close, open() {
      active = true; root.classList.remove('hidden');
      $('report-compose').classList.add('hidden');
      $('report-dialog').classList.remove('composing');
      if (selectedId) backToBoard();
      $('report-dialog').focus(); refresh();
    } };
  }
  const exported = { LANES, Store, laneOf, progress, matches, components, mount };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  global.OpenPetRequests = exported;
})(typeof window === 'undefined' ? globalThis : window);
