# Themes

The app's colours come from one set of semantic tokens: `server/themes.css` (CSS custom properties),
`server/theme-data.js` (the same data) and `server/theme-controller.js` (resolution, system appearance,
validated preferences). `server/theme-app.js` applies them on every page.

- **Presets:** Cobalt (default), Golden, Meadow. **Appearance:** System (default), Light, Dark.
  Chosen per phone in Settings → Appearance and kept in `localStorage` (`openpetcam.theme.preferences.v1`); nothing is
  sent to the server. A tiny inline script in each page's `<head>` sets `data-opc-theme` / `data-opc-mode` before first
  paint, so there is no flash. Without JavaScript, the CSS alone follows the system appearance.
- **Roles, not colours.** Pages map their own names onto the tokens (`--bg`→`--opc-canvas`, `--card`→`--opc-surface`,
  `--accent`→`--opc-accent-fill` with `--opc-on-accent` for its label, `--accent-fg` for accent-coloured text, …).
  Button backgrounds and accent text are separate on purpose; don't collapse them into one variable.
- **Pet states are shared meanings**, the same in every preset: resting = lavender, active = green, bark = amber,
  not in view = slate. Pair them with names or icons, never colour alone.
- **The camera picture is not themed.** Anything drawn over video (status pill, LIVE, tracking box, Find button, rail
  buttons' glass) uses the invariant `--opc-video-*` tokens and the scrim `--opc-video-scrim` behind white text. The
  tracking box is the fixed gold with a dark halo.
- **Contrast:** the token set was checked at 4.5:1 for text and 3:1 for meaningful borders and focus rings in all six
  preset/mode pairs. That covers the tokens, not every component: check new UI in Light and Dark before shipping.

When editing a token, change `theme-data.js` and `themes.css` together and recheck the affected pairs.
