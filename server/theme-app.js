// Applies the owner's chosen theme on every OpenPetCam page (themes.css + theme-controller.js).
// The inline snippet in each page's <head> sets data-opc-theme / data-opc-mode before first paint;
// this module then resolves the full token set, follows the system appearance, keeps the browser's
// theme-color in step, and repaints when another page (Settings, or the Agent tab's frame) changes it.
import { applyTheme, loadPreferences, savePreferences, STORAGE_KEY } from "/theme-controller.js";

const root = document.documentElement;
function syncMeta() {
  const canvas = getComputedStyle(root).getPropertyValue("--opc-canvas").trim();
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta && canvas) meta.setAttribute("content", canvas);
}
root.addEventListener("openpetcamthemechange", syncMeta);

export function paint(prefs = loadPreferences()) {
  const r = applyTheme(prefs);
  syncMeta();
  return r;
}
export function choose(change) {
  const prefs = savePreferences({ ...loadPreferences(), ...change }).preferences;
  return paint(prefs);
}
export { loadPreferences };

paint();
addEventListener("storage", (e) => { if (e.key === STORAGE_KEY) paint(); });
