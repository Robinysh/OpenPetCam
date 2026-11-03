// Generated from themes.json. Keep this file and themes.json in sync.
function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}
export const themeData = deepFreeze({
  "version": "1.0.0",
  "defaultPreferences": {
    "theme": "cobalt",
    "appearance": "system"
  },
  "themes": {
    "cobalt": {
      "label": "Cobalt",
      "light": {
        "canvas": "#F6F8FF",
        "surface": "#FFFFFF",
        "surfaceRaised": "#EDF1FC",
        "textPrimary": "#202332",
        "textSecondary": "#4C586F",
        "textMuted": "#5F6B80",
        "borderSubtle": "#D9DFEE",
        "controlBorder": "#74829B",
        "accentFill": "#355BEB",
        "accentForeground": "#355BEB",
        "onAccent": "#FFFFFF",
        "accentSubtle": "#E6ECFF",
        "accentHover": "#2D4FD5",
        "accentPressed": "#2542BC",
        "focusRing": "#355BEB"
      },
      "dark": {
        "canvas": "#0D1222",
        "surface": "#171F34",
        "surfaceRaised": "#222E47",
        "textPrimary": "#F3F5FF",
        "textSecondary": "#C0CAE0",
        "textMuted": "#A0ADC8",
        "borderSubtle": "#35425D",
        "controlBorder": "#7484A3",
        "accentFill": "#91A7FF",
        "accentForeground": "#A9BAFF",
        "onAccent": "#10172B",
        "accentSubtle": "#26345A",
        "accentHover": "#A0B3FF",
        "accentPressed": "#B0C0FF",
        "focusRing": "#B6C5FF"
      }
    },
    "golden": {
      "label": "Golden",
      "light": {
        "canvas": "#FFFAF0",
        "surface": "#FFFFFF",
        "surfaceRaised": "#F5EBD6",
        "textPrimary": "#2A2418",
        "textSecondary": "#62553D",
        "textMuted": "#74674F",
        "borderSubtle": "#E3D8BE",
        "controlBorder": "#8C7952",
        "accentFill": "#F3CE63",
        "accentForeground": "#805900",
        "onAccent": "#2A2107",
        "accentSubtle": "#FFF0C2",
        "accentHover": "#E9BE45",
        "accentPressed": "#DBAE31",
        "focusRing": "#805900"
      },
      "dark": {
        "canvas": "#17140E",
        "surface": "#242017",
        "surfaceRaised": "#342D1F",
        "textPrimary": "#FFF8E9",
        "textSecondary": "#D7CBAF",
        "textMuted": "#BFB08E",
        "borderSubtle": "#4B432F",
        "controlBorder": "#938362",
        "accentFill": "#F3CE63",
        "accentForeground": "#F3CE63",
        "onAccent": "#2A2107",
        "accentSubtle": "#46381A",
        "accentHover": "#F7D779",
        "accentPressed": "#FAE192",
        "focusRing": "#F3CE63"
      }
    },
    "meadow": {
      "label": "Meadow",
      "light": {
        "canvas": "#F1F8F5",
        "surface": "#FFFFFF",
        "surfaceRaised": "#E3F1EA",
        "textPrimary": "#173128",
        "textSecondary": "#466456",
        "textMuted": "#526B5F",
        "borderSubtle": "#CADFD3",
        "controlBorder": "#668473",
        "accentFill": "#106B58",
        "accentForeground": "#106B58",
        "onAccent": "#FFFFFF",
        "accentSubtle": "#DDF0E6",
        "accentHover": "#0D5E4D",
        "accentPressed": "#0A5142",
        "focusRing": "#106B58"
      },
      "dark": {
        "canvas": "#0D1815",
        "surface": "#172823",
        "surfaceRaised": "#243C32",
        "textPrimary": "#EFFAF5",
        "textSecondary": "#BBD3C6",
        "textMuted": "#A3BDAE",
        "borderSubtle": "#365447",
        "controlBorder": "#699580",
        "accentFill": "#7DD6AE",
        "accentForeground": "#94E2BF",
        "onAccent": "#10271D",
        "accentSubtle": "#264C3B",
        "accentHover": "#91DEBB",
        "accentPressed": "#A4E7C9",
        "focusRing": "#A9EACD"
      }
    }
  },
  "semantic": {
    "light": {
      "resting": "#6F54B3",
      "active": "#16734F",
      "bark": "#9A5800",
      "unseen": "#566578",
      "warning": "#825000",
      "warningSurface": "#FFF2D3",
      "onWarning": "#643D00",
      "error": "#B4233C",
      "errorSurface": "#FFF0F2",
      "onError": "#921A2D",
      "success": "#16734F"
    },
    "dark": {
      "resting": "#B9A6F0",
      "active": "#66D5A0",
      "bark": "#F5B357",
      "unseen": "#9DAFC5",
      "warning": "#F5B357",
      "warningSurface": "#392A10",
      "onWarning": "#FFD89A",
      "error": "#FF879A",
      "errorSurface": "#3B1B25",
      "onError": "#FFD2DA",
      "success": "#66D5A0"
    }
  },
  "video": {
    "scrim": "rgba(10,14,23,0.82)",
    "onVideo": "#FFFFFF",
    "live": "#FF6B6B",
    "tracking": "#F3CE63",
    "resting": "#B9A6F0",
    "active": "#66D5A0",
    "unseen": "#9DAFC5"
  }
});
