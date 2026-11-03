// OpenPetCam website analytics (PostHog project "OpenPetCam").
// Cookieless: nothing is stored in cookies or local storage, IPs are dropped by the project, no session replay.
// Tracks what tells us whether the site works: page views, the film, the install command, GitHub clicks.
(function () {
  if (!/(^|\.)github\.io$/.test(location.hostname)) return;     // local previews never pollute the numbers
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled getFeatureFlag onFeatureFlags".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
  posthog.init("phc_tJrDZeZgJvtKPqvQgv789UXc3HBqKuaFn6rBQVTCFj7M", {
    api_host: "https://us.i.posthog.com",
    defaults: "2026-08-30",
    cookieless_mode: "always",
    disable_session_recording: true,
    capture_pageview: true,
    capture_pageleave: true,
  });
  const q = new URLSearchParams(location.search);
  if (q.get("ref")) posthog.register({ ref: q.get("ref") });
})();

function track(event, props) { try { window.posthog && posthog.capture(event, props || {}); } catch (e) {} }

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href]");
  if (a && /github\.com/.test(a.href)) {
    track("github_click", { target: a.dataset.track || a.href.replace("https://github.com/", ""), section: (a.closest("[data-section]") || {}).dataset?.section });
  }
  const c = e.target.closest("[data-copy]");
  if (c) {
    const text = document.querySelector(c.dataset.copy).innerText.trim();
    navigator.clipboard && navigator.clipboard.writeText(text).then(() => {
      c.textContent = "Copied"; setTimeout(() => (c.textContent = "Copy"), 1600);
    });
    track("install_copied", { section: (c.closest("[data-section]") || {}).dataset?.section });
  }
});

document.querySelectorAll("video[data-track]").forEach((v) => {
  const marks = new Set();
  v.addEventListener("play", () => track("film_play", { at: Math.round(v.currentTime) }));
  v.addEventListener("ended", () => track("film_complete"));
  v.addEventListener("timeupdate", () => {
    const p = Math.floor((v.currentTime / (v.duration || 61)) * 4) * 25;
    if (p > 0 && p < 100 && !marks.has(p)) { marks.add(p); track("film_progress", { percent: p }); }
  });
});

document.querySelectorAll("details[data-faq]").forEach((d) =>
  d.addEventListener("toggle", () => d.open && track("faq_open", { question: d.querySelector("summary").innerText.slice(0, 80) })));
