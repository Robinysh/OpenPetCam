import { themeData } from './theme-data.js';

export { themeData };
export const STORAGE_KEY = 'openpetcam.theme.preferences.v1';
const subscriptions = new WeakMap();
const APPEARANCES = new Set(['system', 'light', 'dark']);
const THEME_NAMES = new Set(Object.keys(themeData.themes));

function normalizeHex(value) {
  if (typeof value !== 'string') return null;
  const h = value.trim();
  return /^#[0-9a-f]{6}$/i.test(h) ? h.toUpperCase() : null;
}
function rgb(hex) {
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
}
function hex(channels) {
  return '#' + channels.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')).join('').toUpperCase();
}
function mix(a, b, amount) {
  const aa = rgb(a), bb = rgb(b);
  return hex(aa.map((v, i) => v + (bb[i] - v) * amount));
}
function luminance(color) {
  const c = rgb(color).map(v => v / 255).map(v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
export function contrastRatio(a, b) {
  const x = luminance(a), y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
function minContrast(color, backgrounds) {
  return Math.min(...backgrounds.map(bg => contrastRatio(color, bg)));
}
function readableForeground(seed, backgrounds, mode, minimum = 4.5) {
  if (minContrast(seed, backgrounds) >= minimum) return seed;
  const preferred = mode === 'dark' ? '#FFFFFF' : '#000000';
  // Check actual 8-bit output colors, without rounding the contrast threshold.
  for (let step = 1; step <= 255; step++) {
    const candidate = mix(seed, preferred, step / 255);
    if (minContrast(candidate, backgrounds) >= minimum) return candidate;
  }
  // All shipped surfaces in a mode share a lightness family. This fallback
  // protects the API if future data introduces a mismatched custom surface.
  const alternate = preferred === '#FFFFFF' ? '#000000' : '#FFFFFF';
  if (minContrast(alternate, backgrounds) >= minimum) return alternate;
  throw new RangeError('No foreground meets the requested contrast on all theme surfaces.');
}
function accessibleAccent(seed, colors, mode) {
  const accentFill = seed;
  const preferred = ['#FFFFFF', '#10172B'].sort((a, b) => contrastRatio(b, accentFill) - contrastRatio(a, accentFill));
  // Pure black is a fallback for middle-luminance seeds where branded ink
  // and white do not quite meet 4.5:1. At least black or white always does.
  const onAccent = contrastRatio(preferred[0], accentFill) >= 4.5 ? preferred[0] : '#000000';
  const lightText = luminance(onAccent) > luminance(accentFill);
  const interactionTarget = lightText ? '#000000' : '#FFFFFF';
  const accentHover = mix(accentFill, interactionTarget, 0.08);
  const accentPressed = mix(accentFill, interactionTarget, 0.16);
  const accentSubtle = mix(colors.surface, seed, mode === 'dark' ? 0.16 : 0.12);
  const backgrounds = [colors.canvas, colors.surface, colors.surfaceRaised, accentSubtle];
  const accentForeground = readableForeground(seed, backgrounds, mode, 4.5);
  return { accentFill, onAccent, accentForeground, accentSubtle, accentHover, accentPressed, focusRing: accentForeground };
}

export function sanitizePreferences(value = {}) {
  const input = value && typeof value === 'object' ? value : {};
  return {
    theme: THEME_NAMES.has(input.theme) ? input.theme : themeData.defaultPreferences.theme,
    appearance: APPEARANCES.has(input.appearance) ? input.appearance : themeData.defaultPreferences.appearance,
    customAccent: normalizeHex(input.customAccent)
  };
}

/** Resolve data without reading the DOM or mutating the canonical theme data.
 * Invalid theme/mode values safely fall back to cobalt/light.
 * customAccent affects only action/selection/focus tokens, never semantic/video.
 */
export function resolveTheme(theme = 'cobalt', mode = 'light', customAccent = null) {
  const resolvedTheme = THEME_NAMES.has(theme) ? theme : 'cobalt';
  const resolvedMode = mode === 'dark' ? 'dark' : 'light';
  const seed = normalizeHex(customAccent);
  const colors = { ...themeData.themes[resolvedTheme][resolvedMode] };
  if (seed) Object.assign(colors, accessibleAccent(seed, colors, resolvedMode));
  return {
    theme: resolvedTheme,
    label: themeData.themes[resolvedTheme].label,
    mode: resolvedMode,
    customAccent: seed,
    colors,
    semantic: { ...themeData.semantic[resolvedMode] },
    video: { ...themeData.video }
  };
}

function kebab(name) { return name.replace(/[A-Z]/g, m => '-' + m.toLowerCase()); }
function systemQuery() {
  return typeof globalThis.matchMedia === 'function' ? globalThis.matchMedia('(prefers-color-scheme: dark)') : null;
}

/** Apply tokens and automatically track system appearance when requested.
 * Reapplying to the same element removes its previous media listener.
 * Returns the initially resolved data and dispose() to release the listener.
 * Does not persist preferences; call savePreferences explicitly if desired.
 */
export function applyTheme(preferences = {}, element = globalThis.document?.documentElement) {
  if (!element?.style?.setProperty || typeof element.setAttribute !== 'function') {
    throw new TypeError('applyTheme requires a DOM element with a style declaration.');
  }
  subscriptions.get(element)?.();
  const safe = sanitizePreferences(preferences);
  const media = systemQuery();
  function paint() {
    const mode = safe.appearance === 'system' ? (media?.matches ? 'dark' : 'light') : safe.appearance;
    const resolved = resolveTheme(safe.theme, mode, safe.customAccent);
    for (const [key, value] of Object.entries({ ...resolved.colors, ...resolved.semantic })) {
      element.style.setProperty('--opc-' + kebab(key), value);
    }
    for (const [key, value] of Object.entries(resolved.video)) {
      element.style.setProperty('--opc-video-' + kebab(key), value);
    }
    element.setAttribute('data-opc-theme', resolved.theme);
    element.setAttribute('data-opc-mode', resolved.mode);
    element.setAttribute('data-opc-appearance', safe.appearance);
    element.style.colorScheme = resolved.mode;
    const EventClass = element.ownerDocument?.defaultView?.CustomEvent ?? globalThis.CustomEvent;
    if (typeof EventClass === 'function' && typeof element.dispatchEvent === 'function') {
      element.dispatchEvent(new EventClass('openpetcamthemechange', {
        bubbles: true,
        detail: { theme: resolved.theme, mode: resolved.mode, appearance: safe.appearance, customAccent: safe.customAccent }
      }));
    }
    return resolved;
  }
  const listener = () => paint();
  let attached = false;
  if (safe.appearance === 'system' && media) {
    if (typeof media.addEventListener === 'function') media.addEventListener('change', listener);
    else if (typeof media.addListener === 'function') media.addListener(listener);
    attached = true;
  }
  function dispose() {
    if (attached && media) {
      if (typeof media.removeEventListener === 'function') media.removeEventListener('change', listener);
      else if (typeof media.removeListener === 'function') media.removeListener(listener);
      attached = false;
    }
    if (subscriptions.get(element) === dispose) subscriptions.delete(element);
  }
  subscriptions.set(element, dispose);
  return { ...paint(), appearance: safe.appearance, dispose };
}

function availableStorage(storage) {
  if (storage !== undefined) return storage;
  try { return globalThis.localStorage ?? null; } catch { return null; }
}
export function loadPreferences(storage) {
  try {
    const raw = availableStorage(storage)?.getItem(STORAGE_KEY);
    return sanitizePreferences(raw ? JSON.parse(raw) : {});
  } catch { return sanitizePreferences({}); }
}
export function savePreferences(preferences, storage) {
  const safe = sanitizePreferences(preferences);
  try {
    const target = availableStorage(storage);
    if (!target?.setItem) return { preferences: safe, persisted: false };
    target.setItem(STORAGE_KEY, JSON.stringify(safe));
    return { preferences: safe, persisted: true };
  } catch { return { preferences: safe, persisted: false }; }
}
