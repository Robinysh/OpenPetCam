# Vendored libraries

Served by the app itself (same origin), so the Room tab loads nothing from a third party.

| File | Source | Licence | sha256 (as downloaded) |
|---|---|---|---|
| `three.module.min.js` | three@0.160.0 `build/three.module.min.js` (jsdelivr) | MIT | `3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6` |
| `OrbitControls.js` | three@0.160.0 `examples/jsm/controls/OrbitControls.js` | MIT | `5a44a9e86a2a0fb11933eed69bc2cd33c76a496854c1aed6ed776efa87d7b064` |

One local edit: `OrbitControls.js` imports `./three.module.min.js` instead of the bare `three` specifier (no import map).
