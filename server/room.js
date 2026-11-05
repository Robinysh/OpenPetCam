// Room tab: your room as measured (the fused surface, textured from the camera, opened up where you look at it from
// behind), where the pet
// is now, and where she spent the day. Loaded only when the tab opens (index.html imports it), so Live stays light.
//
// Designed from what an owner asks of a room, not from the sensors behind it: Where is she? Where does she spend her
// time? When? What is that spot? One look (Day / Night), up to three floor overlays, one day strip to scrub, and a tap
// on any spot. Everything else the model holds (depth points, echoes, sound plots) stays in the developer viewer
// (sim/world/viewer). Data: /api/room (the bundle from sim/world/app_bundle.py) and /api/room/positions.
import * as THREE from "/vendor/three.module.min.js";
import { OrbitControls } from "/vendor/OrbitControls.js";

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const DAY = 86400;
const store = (k, v) => { try { v === undefined ? (v = localStorage.getItem(k)) : localStorage.setItem(k, v); } catch { v = null; } return v; };

export async function mountRoom(root, pet) {
  const name = (pet && pet.name) || "your pet";
  const info = await (await fetch("/api/room", { credentials: "same-origin" })).json().catch(() => ({ available: false }));
  root.innerHTML = "";
  if (!info.available) {
    root.append(el("div", "room-empty", `<h2>No room model yet</h2><p>The Room tab shows your room in 3D, where ${name} is and
      where ${name} spends the day. Build the model on your computer (<code>sim/world</code>, see its README), then
      <code>python3 -m sim.world.app_bundle eval/world</code> and deploy: it is installed on your server only.</p>`));
    return { show() {}, hide() {} };
  }
  // ---------------------------------------------------------------- layout
  const stage = el("div", "room-stage"); root.append(stage);
  const pill = el("div", "room-pill", `<i></i><span>Loading the room…</span>`); root.append(pill);
  const rail = el("div", "room-rail"); root.append(rail);
  const ICON = {
    layers: `<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>`,
    centre: `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg>`,
  };
  const bLayers = el("button", "iconbtn glass", ICON.layers); bLayers.setAttribute("aria-label", "Looks and overlays");
  const bView = el("button", "iconbtn glass word", "3D"); bView.setAttribute("aria-label", "Switch between 3D and the plan from above");
  const bCentre = el("button", "iconbtn glass", ICON.centre); bCentre.setAttribute("aria-label", `Show ${name}`);
  rail.append(bLayers, bView, bCentre);
  const spot = el("div", "room-card hidden"); root.append(spot);
  const dock = el("div", "room-dock"); root.append(dock);
  dock.innerHTML = `<div class="room-dock-head"><button class="iconbtn" data-d="-1" aria-label="Previous day"><svg viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6"/></svg></button>
    <b class="room-day"></b><button class="iconbtn" data-d="1" aria-label="Next day"><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg></button>
    <span class="room-time"></span><button class="chip room-now" type="button">Now</button></div>
    <div class="room-note hidden"></div>
    <canvas class="room-strip" aria-label="The day: drag to see where ${name} was"></canvas>
    <div class="room-strip-axis"><span>0</span><span>6</span><span>12</span><span>18</span><span>24</span></div>`;
  // the views panel sits BESIDE the room (wide screens: a column; phones: a half-height drawer), never over it, so
  // every pick shows at once; the layers button opens and closes it
  const sheet = el("div", "room-panel hidden"); root.append(sheet);
  const LOOKS = [["photo", "Day", "the camera's daytime sweep"], ["night", "Night", "its infrared night picture"]];
  // every view the room model holds, grouped by what it tells you; one shows at a time (each loads when picked)
  const GROUPS = [
    [name, [["heat", "Time spent", `where ${name} was, brighter for longer: up to the time on the strip`]]],
    ["Shape", [["uncertainty", "Unsure areas", "where the model knows least (bright = uncertain)"],
               ["class", "Floor, walls, objects", "what each surface was classified as"],
               ["normals", "Surface direction", "which way each surface faces"],
               ["floor_residual", "Floor check", "learned depth against the floor set by sound (bright = disagrees)"]]],
    ["Light", [["nir_excess", "Materials", "near-infrared minus visible: plants, fabrics and fur glow"],
               ["nir_reflectance", "Infrared reflectance", "the camera's own IR light, corrected for distance"],
               ["light_range", "Daylight change", "how much each surface's light changes over the day"]]],
    ["Sound", [["echo_support", "Echo support", "where the camera's own sound echoes agree with the shape"],
               ["echo_coverage", "Sound coverage", "how many head directions heard each surface"],
               ["source", "Where each part comes from", "grey: seen by the camera; green: the surface above it, measured by sound; violet: floor under the camera, assumed at the height sound measured"]]]];
  const tx = (k) => info.textures?.[k] ? `/api/room/tex?f=${encodeURIComponent(info.textures[k])}` : null;
  sheet.innerHTML = `<div class="room-panel-in" role="region" aria-label="Views of the room"><div class="room-panel-head"><h3>Views</h3>
    <button class="iconbtn room-close" type="button" aria-label="Hide the views"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>
    <div class="room-sec">Look</div><div class="room-tiles">${LOOKS.filter(([k]) => tx(k)).map(([k, t, d]) =>
      `<button class="room-tile" data-look="${k}" title="${d}"><span style="background-image:url(${tx(k)})"></span>${t}</button>`).join("")}</div>
    ${GROUPS.map(([g, items]) => { const its = items.filter(([k]) => k === "heat" || tx(k)); return its.length ? `<div class="room-sec">${g}</div>` + its.map(([k, t, d]) =>
      `<button class="room-over" data-over="${k}"><span class="room-thumb ${k}"${k !== "heat" ? ` data-bg="${tx(k)}"` : ""}></span>
       <span><b>${t}</b><small>${d}</small></span><i class="room-check"></i></button>` + (k === "heat" ? `<div class="seg room-span" role="group" aria-label="Period">
       <button data-span="day">That day</button><button data-span="week">Last 7 days</button></div>` : "")).join("") : ""; }).join("")}
    <p class="room-foot">Built ${info.built}. The model lives on your server only.</p></div>`;

  // ---------------------------------------------------------------- state
  const S = { span: store("room.span") || "day", week: [], look: store("room.look") || "photo", over: (store("room.over") === "dwell" ? "heat" : store("room.over")) || "heat", view: store("room.view") || "3d",
              day0: startOfDay(Date.now() / 1000), T: null, rows: [], live: true };
  let raf = 0, visible = false;           // render on demand; declared before any texture can finish loading
  // ---------------------------------------------------------------- 3D
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  stage.append(renderer.domElement);
  const scene = new THREE.Scene();
  const bg = () => { scene.background = new THREE.Color(css("--bg") || "#111"); };
  bg();
  const world = new THREE.Group(); world.scale.x = -1; scene.add(world);       // the model's frame is left-handed
  const cam = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49; controls.minDistance = 1.5; controls.maxDistance = 16;
  const H = info.floor_height_m;
  const loader = new THREE.TextureLoader();
  const tex = (k) => { const u = tx(k); if (!u) return null; const t = loader.load(u, () => draw()); t.colorSpace = THREE.SRGBColorSpace; return t; };
  const T = new Proxy({}, { get: (c, k) => (k in c ? c[k] : (c[k] = tex(k))) });   // loaded on first use
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1); blank.needsUpdate = true;
  const U = { pano: { value: blank }, over: { value: blank }, overOn: { value: 0 }, cut: { value: 0 }, mute: { value: 0 },
              clay: { value: new THREE.Color("#c9c6c0") }, light: { value: new THREE.Vector3(0.45, 0.8, 0.35).normalize() } };
  const vert = `varying vec3 wp; void main(){ vec4 w = modelMatrix * vec4(position,1.); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
  const frag = `uniform sampler2D pano, over; uniform float overOn, cut, mute; uniform vec3 clay, light; varying vec3 wp;
  void main(){
    vec3 d = normalize(vec3(-wp.x, wp.y, wp.z));
    vec2 uv = vec2(atan(d.x, d.z) / 6.2831853 + 0.5, 0.5 + asin(clamp(d.y,-1.,1.)) / 3.1415927);
    vec3 p = texture2D(pano, uv).rgb; float seen = step(0.02, p.r + p.g + p.b);
    vec3 fn = normalize(cross(dFdx(wp), dFdy(wp))); float lens = dot(fn, normalize(-wp));
    if (cut > 0.5 && lens < -0.02) discard;                       // seen from behind: cut away (dollhouse)
    float sh = 0.64 + 0.36 * (0.5 + 0.5 * dot(fn, normalize(vec3(-light.x, light.y, light.z))));
    vec3 col = mix(clay * sh, p * mix(1.0, sh, 0.3), seen);
    if (lens < 0.12) col = clay * sh * 0.93;                       // the lens only grazed it: untextured, honestly
    if (mute > 0.5) col = mix(vec3(dot(col, vec3(0.3, 0.59, 0.11))), col, 0.35) * 0.62;   // a quiet base under the heat
    if (overOn > 0.5) { vec3 o = texture2D(over, uv).rgb; float so = step(0.03, o.r + o.g + o.b); col = mix(col * 0.55, o, so * 0.85); }
    gl_FragColor = vec4(col, 1.);
    #include <colorspace_fragment>
  }`;
  const mat = (cut) => new THREE.ShaderMaterial({ uniforms: { ...U, cut: { value: cut } }, vertexShader: vert, fragmentShader: frag, side: THREE.DoubleSide });
  const solidMat = mat(0), cutMat = mat(1);
  // The measured shape (accuracy first): the fused surface, one range per viewing direction (millimetres), meshed
  // with tears at depth jumps, cut away where you look at it from behind. No fitted planes are drawn: they are a
  // simplification of this, and they misplace whatever they leave out. Furniture boxes only name places.
  const [RW, RH] = info.range_shape || [0, 0];
  let R = new Float32Array(0);
  if (RW && tx("range")) {
    const u = new Uint16Array(await (await fetch(tx("range"), { credentials: "same-origin" })).arrayBuffer());
    if (u.length === RW * RH) R = Float32Array.from(u, (v) => v / 1000);
  }
  const pos = new Float32Array(RW * RH * 3);
  for (let j = 0; j < RH; j++) for (let i = 0; i < RW; i++) {
    const az = ((i + 0.5) / RW * 360 - 180) * Math.PI / 180, el = (90 - (j + 0.5) / RH * 180) * Math.PI / 180, r = R[j * RW + i] || 0, k = (j * RW + i) * 3;
    pos[k] = Math.cos(el) * Math.sin(az) * r; pos[k + 1] = Math.sin(el) * r; pos[k + 2] = Math.cos(el) * Math.cos(az) * r;
  }
  const va = new THREE.Vector3(), vb = new THREE.Vector3(), vc = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), dd = new THREE.Vector3();
  const good = (a, b, c) => {
    const ra = R[a], rb = R[b], rc = R[c]; if (!ra || !rb || !rc) return false;
    const mx = Math.max(ra, rb, rc), mn = Math.min(ra, rb, rc); if (!(mx / mn < 1.18 || mx - mn < 0.05)) return false;   // a depth jump: tear
    va.fromArray(pos, a * 3); vb.fromArray(pos, b * 3); vc.fromArray(pos, c * 3);
    const n = e1.subVectors(vb, va).cross(e2.subVectors(vc, va)), len = n.length(); if (!len) return false;
    return Math.abs(n.dot(dd.copy(va).add(vb).add(vc).normalize())) / len > 0.12;          // edge-on to the lens: a sliver
  };
  const tris = [], use = new Uint8Array(RW * RH), idx = [];
  for (let j = 0; j < RH - 1; j++) for (let i = 0; i < RW; i++) {
    const i2 = (i + 1) % RW, a = j * RW + i, b = j * RW + i2, c = (j + 1) * RW + i, d = (j + 1) * RW + i2;
    if (good(a, c, b)) { tris.push(a, c, b); use[a]++; use[b]++; use[c]++; }
    if (good(b, c, d)) { tris.push(b, c, d); use[b]++; use[c]++; use[d]++; }
  }
  for (let t = 0; t < tris.length; t += 3) if (use[tris[t]] >= 4 && use[tris[t + 1]] >= 4 && use[tris[t + 2]] >= 4) idx.push(tris[t], tris[t + 1], tris[t + 2]);
  const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.BufferAttribute(pos, 3)); sg.setIndex(idx);
  const surface = new THREE.Mesh(sg, cutMat); world.add(surface);
  // the measured range in a direction; a floor cell counts as floor only where nothing nearer blocks the lens's view
  const rangeAt = (x, y, z) => {
    const r = Math.hypot(x, y, z), az = Math.atan2(x, z) * 180 / Math.PI, el = Math.asin(y / r) * 180 / Math.PI;
    return R[Math.min(RH - 1, Math.floor((90 - el) / 180 * RH)) * RW + (Math.floor((az + 180) / 360 * RW) % RW)] || 0;
  };
  const onFloor = (x, z) => { const r = Math.hypot(x, H, z), rr = rangeAt(x, -H, z); return rr > 0 && rr >= r * 0.92 && r < 8; };
  const objects = info.objects.map((o) => ({ ...o }));        // named places (not drawn: the surface already shows them)
  // the camera (a small dot at the lens)
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshBasicMaterial({ color: css("--accent") || "#7aa7ff" })); world.add(lens);
  // where the camera points: a view cone from the lens out to the measured surface its frame corners land on
  // (pose from the watcher's track at the strip's time; the calibrated camera with the mount's drift correction)
  const C = info.calib, OFF = info.mount_offset_deg || [0, 0, 0], D2R = Math.PI / 180;
  const mul = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const rx = (a) => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const ry = (a) => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  const rz = (a) => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
  function camRay(pan, tilt, u, v) {                       // roomcam.Camera.ray without the lens distortion term
    const Rm = [ry((1850 - pan) * C.deg_per_step_pan * D2R), rx(-(tilt - C.level_tilt) * C.deg_per_step_tilt * D2R), rz(C.roll_deg * D2R),
                ry(OFF[1] * D2R), rx(OFF[0] * D2R)].reduce(mul);
    const c = [(u * 1920 - 960) / C.focal_px, -(v * 1080 - 540) / C.focal_px, 1];
    const w = [0, 1, 2].map((i) => Rm[i][0] * c[0] + Rm[i][1] * c[1] + Rm[i][2] * c[2]), n = Math.hypot(...w);
    return w.map((x) => x / n);
  }
  const accent = new THREE.Color(css("--accent") || "#7aa7ff");
  const coneGeo = new THREE.BufferGeometry(), coneLines = new THREE.BufferGeometry();
  const cone = new THREE.Mesh(coneGeo, new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
  const coneEdge = new THREE.LineSegments(coneLines, new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.85 }));
  cone.renderOrder = coneEdge.renderOrder = 3; world.add(cone, coneEdge);
  const axis = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.45, accent.getHex(), 0.12, 0.08); world.add(axis);
  function camRayFor(T) { let pose = null; for (const p of S.poses || []) { if (p[0] <= T) pose = p; else break; } return pose ? camRay(pose[1], pose[2], 0.5, 0.5) : []; }
  function showCamera(pose) {
    cone.visible = coneEdge.visible = axis.visible = !!pose;
    if (!pose) return;
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => {
      const d = camRay(pose[1], pose[2], u, v), r = Math.min(rangeAt(d[0] * 5, d[1] * 5, d[2] * 5) || 3, 6);
      return d.map((x) => x * r);
    });
    const P = []; for (let k = 0; k < 4; k++) P.push(0, 0, 0, ...corners[k], ...corners[(k + 1) % 4]);
    coneGeo.setAttribute("position", new THREE.Float32BufferAttribute(P, 3)); coneGeo.computeBoundingSphere();
    const E = []; for (let k = 0; k < 4; k++) E.push(0, 0, 0, ...corners[k], ...corners[k], ...corners[(k + 1) % 4]);
    coneLines.setAttribute("position", new THREE.Float32BufferAttribute(E, 3)); coneLines.computeBoundingSphere();
    axis.setDirection(new THREE.Vector3(...camRay(pose[1], pose[2], 0.5, 0.5)));
  }
  // time-spent heat on the floor
  const HC = 0.1, HN = 140, HALF = HN * HC / 2, heatData = new Uint8Array(HN * HN * 4);
  const heatTex = new THREE.DataTexture(heatData, HN, HN); heatTex.magFilter = THREE.LinearFilter; heatTex.needsUpdate = true;
  const heat = new THREE.Mesh(new THREE.PlaneGeometry(HN * HC, HN * HC), new THREE.MeshBasicMaterial({ map: heatTex, transparent: true, depthWrite: false }));
  heat.rotation.x = -Math.PI / 2; heat.position.y = -H + 0.012; heat.renderOrder = 2; world.add(heat);
  // the pet
  const petCol = new THREE.Color(css("--viz-active") || "#ff8a5b");
  const dot = new THREE.Group(); world.add(dot);
  dot.add(new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.05, 32), new THREE.MeshBasicMaterial({ color: petCol })));
  dot.add(new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.04, 32), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  const halo = new THREE.Mesh(new THREE.RingGeometry(0.24, 0.31, 48), new THREE.MeshBasicMaterial({ color: petCol, transparent: true, side: THREE.DoubleSide }));
  halo.rotation.x = -Math.PI / 2; dot.add(halo); dot.visible = false;

  // ---------------------------------------------------------------- behaviour
  const label = (o) => o.label.replace(/\s*\(.*?\)\s*/g, "").replace(/,.*$/, "").trim();
  function placeOf(x, z) {
    let best = null, bd = 0.6;
    for (const o of objects) {
      if (o.kind === "thin") continue;                       // a screen is not a place she can be
      const dx = x - o.center[0], dz = z - o.center[2], c = Math.cos(o.yaw_deg * Math.PI / 180), s = Math.sin(o.yaw_deg * Math.PI / 180);
      const lx = Math.abs(dx * c - dz * s) - o.size[0] / 2, lz = Math.abs(dx * s + dz * c) - o.size[2] / 2;
      const dd = Math.hypot(Math.max(lx, 0), Math.max(lz, 0));
      if (dd < bd) { bd = dd; best = o; }
    }
    return best ? (bd < 0.05 ? `on the ${label(best)}` : `by the ${label(best)}`) : "on the open floor";
  }
  const fmtT = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const fmtDur = (s) => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  function startOfDay(t) { const d = new Date(t * 1000); d.setHours(0, 0, 0, 0); return d.getTime() / 1000; }
  const dayLabel = (d0) => { const today = startOfDay(Date.now() / 1000); return d0 === today ? "Today" : d0 === today - DAY ? "Yesterday" : new Date(d0 * 1000).toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }); };

  async function loadDay() {
    const until = Math.min(S.day0 + DAY, Date.now() / 1000);
    try {
      const r = await (await fetch(`/api/room/positions?since=${S.day0}&until=${until}`, { credentials: "same-origin" })).json();
      S.rows = r.rows || []; S.poses = r.poses || [];
    } catch { S.rows = []; }
    if (!S.rows.length && S.lastDay == null) {                // find the last day she was seen (up to a week back)
      try {
        const r = await (await fetch(`/api/room/positions?since=${S.day0 - 3 * DAY}&until=${S.day0}`, { credentials: "same-origin" })).json();
        const last = (r.rows || []).at(-1); S.lastDay = last ? startOfDay(last[0]) : -1;
      } catch { S.lastDay = -1; }
    }
    if (S.live || S.T == null || S.T > until || S.T < S.day0) S.T = until;
    if (S.span === "week") await loadWeek();
    apply();
  }
  async function loadWeek() {                                 // the 6 days before the strip's day, then that day itself
    const parts = [];
    for (let a = S.day0 - 6 * DAY; a < S.day0; a += 3 * DAY) {
      try { parts.push(...((await (await fetch(`/api/room/positions?since=${a}&until=${Math.min(a + 3 * DAY, S.day0)}`, { credentials: "same-origin" })).json()).rows || [])); } catch { /* a gap stays a gap */ }
    }
    S.week = parts;
  }
  function buildHeat() {
    const g = new Float32Array(HN * HN);
    for (const r of (S.span === "week" ? S.week.concat(S.rows) : S.rows)) { if (r[0] > S.T || r[5]) continue; const i = Math.floor((r[1] + HALF) / HC), j = Math.floor((r[2] + HALF) / HC); if (i >= 0 && i < HN && j >= 0 && j < HN) g[j * HN + i] += r[4]; }
    const k = [0.06, 0.24, 0.4, 0.24, 0.06], t = new Float32Array(HN * HN), b = new Float32Array(HN * HN);
    for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) { let s = 0; for (let d = -2; d <= 2; d++) { const ii = i + d; if (ii >= 0 && ii < HN) s += g[j * HN + ii] * k[d + 2]; } t[j * HN + i] = s; }
    for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) { let s = 0; for (let d = -2; d <= 2; d++) { const jj = j + d; if (jj >= 0 && jj < HN) s += t[jj * HN + i] * k[d + 2]; } b[j * HN + i] = s; }
    let mx = 0; for (const v of b) mx = Math.max(mx, v);
    const stops = [[0, [92, 38, 20]], [0.35, [205, 82, 44]], [0.7, [255, 146, 98]], [1, [255, 236, 220]]];
    for (let p = 0; p < HN * HN; p++) {
      const row = p / HN | 0, col = p % HN, o = ((HN - 1 - row) * HN + col) * 4;
      const v = mx ? Math.sqrt(b[p] / mx) : 0;
      let c = stops[0][1]; for (let s = 1; s < stops.length; s++) if (v <= stops[s][0]) { const a = stops[s - 1], z = stops[s], f = (v - a[0]) / (z[0] - a[0]); c = a[1].map((x, q) => x + (z[1][q] - x) * f); break; }
      heatData[o] = c[0]; heatData[o + 1] = c[1]; heatData[o + 2] = c[2];
      heatData[o + 3] = v < 0.03 || !onFloor(col * HC - HALF + HC / 2, row * HC - HALF + HC / 2) ? 0 : Math.min(245, 110 + 160 * v);
    }
    heatTex.needsUpdate = true;
  }
  function apply() {
    U.pano.value = T[S.look] || T.photo || blank;
    const ov = S.over && S.over !== "heat" && S.over !== "none" ? T[S.over] : null;
    U.over.value = ov || blank; U.overOn.value = ov ? 1 : 0;
    for (const m of [solidMat, cutMat]) { m.uniforms.pano.value = U.pano.value; m.uniforms.over.value = U.over.value; m.uniforms.overOn.value = U.overOn.value; m.uniforms.mute.value = S.over === "heat" ? 1 : 0; }
    heat.visible = S.over === "heat"; if (heat.visible) buildHeat();
    // where the camera pointed at the playhead
    let pose = null; for (const p of S.poses || []) { if (p[0] <= S.T) pose = p; else break; }
    showCamera(pose);
    // where she is (or was, at the playhead)
    let last = null; for (const r of S.rows) { if (r[0] <= S.T) last = r; else break; }
    const ago = last ? S.T - last[0] : Infinity;
    dot.visible = !!last && ago < 6 * 3600; if (last) dot.position.set(last[1], -H + 0.02, last[2]);
    dot.children[0].material.opacity = 1; halo.material.opacity = ago < 120 ? 1 : 0.35;
    const live = S.live && Math.abs(Date.now() / 1000 - S.T) < 90;
    pill.className = `room-pill ${last && ago < 120 ? "here" : ""}`;
    pill.querySelector("span").textContent = !last ? `${name} not seen ${S.day0 === startOfDay(Date.now() / 1000) ? "today" : "that day"}`
      : ago < 120 ? `${name} · ${placeOf(last[1], last[2])}${live ? " · now" : ` · ${fmtT(last[0])}`}`
      : live ? `${name} · ${placeOf(last[1], last[2])} · ${fmtDur(ago)} ago` : `${name} · ${placeOf(last[1], last[2])} · ${fmtT(last[0])}`;
    dock.querySelector(".room-day").textContent = dayLabel(S.day0);
    dock.querySelector(".room-time").textContent = live ? "now" : S.T >= S.day0 + DAY - 60 ? "end of day" : fmtT(S.T);
    dock.querySelector(".room-now").classList.toggle("hidden", live);
    dock.querySelector('[data-d="1"]').disabled = S.day0 >= startOfDay(Date.now() / 1000);
    sheet.querySelectorAll("[data-look]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.look === S.look));
    sheet.querySelectorAll("[data-over]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.over === S.over));
    sheet.querySelectorAll("[data-span]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.span === S.span));
    const span = sheet.querySelector(".room-span"); if (span) span.hidden = S.over !== "heat";
    bView.textContent = S.view === "3d" ? "3D" : "Top";
    // nothing to show for this day: say so, and offer the last day she was seen (an empty floor explains nothing)
    const note = dock.querySelector(".room-note"), shown = S.rows.filter((r) => r[0] <= S.T).length;
    note.classList.toggle("hidden", shown > 0);
    if (!shown) {
      const today = S.day0 === startOfDay(Date.now() / 1000);
      note.innerHTML = `${today ? `${name} has not been seen in the room yet today` : `${name} was not seen in the room that day`}${S.lastDay != null && S.lastDay !== S.day0 ? ` · <button class="room-link" type="button">Show ${dayLabel(S.lastDay).toLowerCase()}</button>` : ""}`;
      const b = note.querySelector(".room-link"); if (b) b.onclick = () => { S.day0 = S.lastDay; S.live = false; S.T = null; loadDay(); };
    }
    drawStrip(); draw();
  }
  // day strip: minutes she was seen, in 10-minute bars; the playhead is draggable
  const strip = dock.querySelector(".room-strip");
  function drawStrip() {
    const w = strip.clientWidth, h = strip.clientHeight, r = devicePixelRatio || 1;
    if (!w) return;
    strip.width = w * r; strip.height = h * r; const g = strip.getContext("2d"); g.scale(r, r); g.clearRect(0, 0, w, h);
    const bins = new Float32Array(144); for (const x of S.rows) { const k = Math.floor((x[0] - S.day0) / 600); if (k >= 0 && k < 144) bins[k] += x[4]; }
    const mx = Math.max(60, ...bins);
    g.fillStyle = css("--line") || "#333"; g.fillRect(0, h - 1, w, 1);
    const end = Math.min(1, (Date.now() / 1000 - S.day0) / DAY);
    g.fillStyle = css("--card2") || "#222"; g.fillRect(end * w, 0, w - end * w, h - 1);           // the future
    g.fillStyle = css("--viz-active") || "#ff8a5b";
    for (let k = 0; k < 144; k++) if (bins[k]) { const bh = Math.max(3, (h - 6) * Math.sqrt(bins[k] / mx)); g.fillRect(k / 144 * w + 0.5, h - 1 - bh, w / 144 - 1, bh); }
    const x = (S.T - S.day0) / DAY * w;
    g.fillStyle = css("--text") || "#fff"; g.fillRect(x - 1, 0, 2, h); g.beginPath(); g.arc(x, 4, 4, 0, 7); g.fill();
  }
  const scrub = (e) => {
    const b = strip.getBoundingClientRect(), f = Math.max(0, Math.min(1, ((e.touches ? e.touches[0].clientX : e.clientX) - b.left) / b.width));
    S.T = Math.min(S.day0 + f * DAY, Date.now() / 1000); S.live = false; apply();
  };
  let dragging = false;
  strip.addEventListener("pointerdown", (e) => { dragging = true; strip.setPointerCapture(e.pointerId); scrub(e); });
  strip.addEventListener("pointermove", (e) => dragging && scrub(e));
  strip.addEventListener("pointerup", () => { dragging = false; });
  dock.querySelector(".room-now").onclick = () => { S.day0 = startOfDay(Date.now() / 1000); S.live = true; loadDay(); };
  dock.querySelectorAll("[data-d]").forEach((b) => b.onclick = () => { S.day0 += +b.dataset.d * DAY; S.live = false; S.T = null; loadDay(); });
  // controls
  function panel(open) {
    if (open) sheet.querySelectorAll("[data-bg]").forEach((e) => { e.style.backgroundImage = `url(${e.dataset.bg})`; e.removeAttribute("data-bg"); });
    sheet.classList.toggle("hidden", !open); root.classList.toggle("panel-open", open);
    bLayers.setAttribute("aria-pressed", String(open)); store("room.panel", open ? "1" : "0");
    requestAnimationFrame(resize);
  }
  bLayers.onclick = () => panel(sheet.classList.contains("hidden"));
  sheet.querySelector(".room-close").onclick = () => panel(false);
  sheet.querySelectorAll("[data-span]").forEach((b) => b.onclick = async () => {
    S.span = b.dataset.span; store("room.span", S.span); S.over = "heat"; store("room.over", "heat");
    if (S.span === "week") await loadWeek(); apply();
  });
  sheet.querySelectorAll("[data-look]").forEach((b) => b.onclick = () => { S.look = b.dataset.look; store("room.look", S.look); apply(); });
  sheet.querySelectorAll("[data-over]").forEach((b) => b.onclick = () => { S.over = S.over === b.dataset.over ? "none" : b.dataset.over; store("room.over", S.over); apply(); });
  // fit the camera to the floor's extent for this screen's shape (a phone in portrait needs to stand much farther back)
  const ext = (() => {
    const xs = [], zs = [];                                       // the measured floor (scene x is the model's -x)
    const cells = new Set();                                      // unique 25 cm cells: near-lens pixels must not outvote the room
    for (let k = 0; k < RW * RH; k++) if (R[k] && pos[k * 3 + 1] < -H + 0.12 && R[k] < 8) {
      const key = `${Math.round(-pos[k * 3] / 0.25)},${Math.round(pos[k * 3 + 2] / 0.25)}`;
      if (!cells.has(key)) { cells.add(key); xs.push(-pos[k * 3]); zs.push(pos[k * 3 + 2]); }
    }
    if (!xs.length) { xs.push(-3, 3); zs.push(-3, 3); }
    const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
    const x0 = q(xs, 0.04), x1 = q(xs, 0.96), z0 = q(zs, 0.04), z1 = q(zs, 0.96);
    return { cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, r: Math.max(x1 - x0, z1 - z0) / 2 + 0.4 };
  })();
  function frame(view) {
    const vfov = cam.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov / 2) * (cam.aspect || 1));
    const d = 1.15 * ext.r / Math.sin(Math.min(vfov, hfov) / 2);
    controls.target.set(ext.cx, -H, ext.cz);                    // scene coordinates (the model's x is mirrored)
    if (view === "top") { cam.position.set(ext.cx, -H + d, ext.cz + 0.01); controls.enableRotate = false; }
    else { const el = 60 * Math.PI / 180, az = -35 * Math.PI / 180;
           cam.position.set(ext.cx + d * Math.cos(el) * Math.sin(az), -H + d * Math.sin(el), ext.cz - d * Math.cos(el) * Math.cos(az)); controls.enableRotate = true; }
    controls.maxDistance = d * 2; controls.update(); draw();
  }
  bView.onclick = () => { S.view = S.view === "3d" ? "top" : "3d"; store("room.view", S.view); frame(S.view); apply(); };
  bCentre.onclick = () => { if (dot.visible) { const p = dot.getWorldPosition(new THREE.Vector3()); controls.target.copy(p); controls.update(); draw(); } else frame(S.view); };
  // tap a spot: what it is and how long she spent there that day
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(); let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
    const b = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1); ray.setFromCamera(ndc, cam);
    const hit = ray.intersectObject(surface, false)[0];
    if (!hit) { spot.classList.add("hidden"); return; }
    const p = world.worldToLocal(hit.point.clone());
    const near = S.rows.filter((r) => Math.hypot(r[1] - p.x, r[2] - p.z) < 0.5 && r[0] <= S.T);
    const total = near.reduce((s, r) => s + r[4], 0);
    spot.innerHTML = `<b>${placeOf(p.x, p.z).replace(/^(on|by) the /, "").replace(/^on the open floor$/, "Open floor")}</b>
      <span>${near.length ? `${name} spent ${fmtDur(total)} here ${dayLabel(S.day0).toLowerCase() === "today" ? "today" : "that day"} · last ${fmtT(near.at(-1)[0])}` : `${name} was not seen here ${S.day0 === startOfDay(Date.now() / 1000) ? "today" : "that day"}`}</span>
      <small>${(Math.hypot(p.x, p.z)).toFixed(1)} m from the camera</small>`;
    spot.classList.remove("hidden");
  });
  // render on demand (battery): while the controls move, when data changes, and the halo's pulse while she is here
  function draw() { if (!raf && visible) raf = requestAnimationFrame(tick); }
  function tick(t) {
    raf = 0;
    const moving = controls.update();
    camLocal.copy(cam.position); world.worldToLocal(camLocal);
    const pulse = dot.visible && halo.material.opacity === 1;
    if (pulse) { const s = 1 + 0.25 * (0.5 + 0.5 * Math.sin(t / 300)); halo.scale.set(s, s, s); }
    renderer.render(scene, cam);
    if (moving || pulse) draw();
  }
  const camLocal = new THREE.Vector3();
  controls.addEventListener("change", draw);
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
    const first = !cam.userData.sized; cam.userData.sized = true;
    renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); if (first) frame(S.view); drawStrip(); draw();
  }
  new ResizeObserver(resize).observe(stage);
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { bg(); drawStrip(); draw(); });
  frame(S.view);
  const wide = matchMedia("(min-width: 900px)").matches;
  panel(store("room.panel") ? store("room.panel") === "1" : wide);     // open by default where there is room beside it
  let poll = 0;
  return {
    show() { visible = true; bg(); resize(); loadDay(); clearInterval(poll); poll = setInterval(() => { if (S.live) loadDay(); }, 30000); },
    hide() { visible = false; clearInterval(poll); },
    state: S, cam, controls, redraw: draw, // for scripted checks
    scrubTo(t) { S.T = Math.min(t, Date.now() / 1000); S.live = false; apply(); },
    probe() {                                                   // what is drawn, in model coordinates
      return { marker: dot.visible ? dot.position.toArray() : null, coneVisible: cone.visible,
               axis: axis.visible ? axis.getWorldDirection(new THREE.Vector3()).toArray() : null,
               axisLocal: axis.visible ? [...camRayFor(S.T)] : null, heatSum: heatData.reduce((a, b) => a + b, 0),
               pill: pill.textContent, floorH: H };
    },
  };
}
