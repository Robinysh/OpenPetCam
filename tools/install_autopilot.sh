#!/usr/bin/env bash
# Schedule tools/autopilot.py nightly on this Mac (launchd). Re-run to update; --remove to uninstall.
# launchd runs a missed job when the Mac wakes, so a sleeping laptop just runs it later. If the Mac
# is off for 36 h the server pushes "nightly check-up hasn't run" to your phone.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=ai.openpetcam.autopilot
PLIST=~/Library/LaunchAgents/$LABEL.plist
if [ "${1:-}" = --remove ]; then for p in "$PLIST" ~/Library/LaunchAgents/ai.openpetcam.reports.plist; do launchctl bootout "gui/$(id -u)" "$p" 2>/dev/null || true; rm -f "$p"; done; echo removed; exit 0; fi
PY=$(command -v python3); CLAUDE=$(command -v claude); TIMEOUT=$(command -v timeout)
[ -n "$CLAUDE" ] && [ -n "$TIMEOUT" ] || { echo "need claude and timeout (brew install coreutils) on PATH"; exit 1; }
"$PY" -c "import onnxruntime, numpy, PIL" || { echo "$PY needs: pip install onnxruntime numpy pillow"; exit 1; }
HOUR=${AUTOPILOT_HOUR:-3}; MIN=${AUTOPILOT_MINUTE:-30}
mkdir -p eval/autopilot ~/Library/LaunchAgents
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$TIMEOUT</string><string>-k</string><string>30</string><string>3h</string>
    <string>$PY</string><string>$PWD/tools/autopilot.py</string><string>nightly</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$(dirname "$CLAUDE"):$(dirname "$TIMEOUT"):$(dirname "$PY"):/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>CLAUDE_CODE_HOOKS_ENABLED</key><string>false</string>
  </dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MIN</integer></dict>
  <key>StandardOutPath</key><string>$PWD/eval/autopilot/launchd.log</string>
  <key>StandardErrorPath</key><string>$PWD/eval/autopilot/launchd.log</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "scheduled $LABEL daily at $(printf %02d:%02d "$HOUR" "$MIN"); run now with: launchctl kickstart gui/$(id -u)/$LABEL"

# The report agent: every 5 minutes, answers "Something wrong?" reports (tools/report_agent.py). A run with
# nothing new is one SSH call. Hard cap 1 h per run; the script holds a lock so runs never overlap.
RLABEL=ai.openpetcam.reports
RPLIST=~/Library/LaunchAgents/$RLABEL.plist
cat > "$RPLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$RLABEL</string>
  <key>ProgramArguments</key><array>
    <string>$TIMEOUT</string><string>-k</string><string>30</string><string>1h</string>
    <string>$PY</string><string>$PWD/tools/report_agent.py</string><string>run</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$(dirname "$CLAUDE"):$(dirname "$TIMEOUT"):$(dirname "$PY"):/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>CLAUDE_CODE_HOOKS_ENABLED</key><string>false</string>
    <key>GIT_TERMINAL_PROMPT</key><string>0</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$PWD/eval/report_agent.launchd.log</string>
  <key>StandardErrorPath</key><string>$PWD/eval/report_agent.launchd.log</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)" "$RPLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$RPLIST"
echo "scheduled $RLABEL every 5 min; run now with: launchctl kickstart gui/$(id -u)/$RLABEL"
