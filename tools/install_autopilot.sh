#!/usr/bin/env bash
# Schedule tools/autopilot.py nightly on this Mac (launchd). Re-run to update; --remove to uninstall.
# launchd runs a missed job when the Mac wakes, so a sleeping laptop just runs it later. If the Mac
# is off for 36 h the server pushes "nightly check-up hasn't run" to your phone.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=ai.openpetcam.autopilot
PLIST=~/Library/LaunchAgents/$LABEL.plist
if [ "${1:-}" = --remove ]; then for p in "$PLIST" ~/Library/LaunchAgents/ai.openpetcam.reports.plist ~/Library/LaunchAgents/ai.openpetcam.agent.*.plist; do launchctl bootout "gui/$(id -u)" "$p" 2>/dev/null || true; rm -f "$p"; done; echo removed; exit 0; fi
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

# The super-agent (tools/superagent.py): tick every 2 min (collect, route, start workers; a run with nothing new is a
# few SSH calls), metrics every 30 min, watchdog every 15 min. It replaces the report agent's own job: reports are one
# of its worker kinds. Each job is hard-capped; the scripts hold locks so runs never overlap.
for spec in "tick:120:30m" "metrics:1800:20m" "watchdog:900:5m"; do
  IFS=: read -r job every cap <<< "$spec"
  JL=ai.openpetcam.agent.$job; JP=~/Library/LaunchAgents/$JL.plist
  cat > "$JP" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$JL</string>
  <key>ProgramArguments</key><array>
    <string>$TIMEOUT</string><string>-k</string><string>30</string><string>$cap</string>
    <string>$PY</string><string>$PWD/tools/superagent.py</string><string>$job</string>
  </array>
  <key>WorkingDirectory</key><string>$PWD</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$(dirname "$CLAUDE"):$(dirname "$TIMEOUT"):$(dirname "$PY"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>CLAUDE_CODE_HOOKS_ENABLED</key><string>false</string>
    <key>GIT_TERMINAL_PROMPT</key><string>0</string>
  </dict>
  <key>StartInterval</key><integer>$every</integer>
  <key>StandardOutPath</key><string>$PWD/eval/superagent/launchd-$job.log</string>
  <key>StandardErrorPath</key><string>$PWD/eval/superagent/launchd-$job.log</string>
</dict></plist>
PL
  mkdir -p eval/superagent
  launchctl bootout "gui/$(id -u)" "$JP" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$JP"
done
# the old standalone report job is now a worker of the super-agent
launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/ai.openpetcam.reports.plist 2>/dev/null || true
rm -f ~/Library/LaunchAgents/ai.openpetcam.reports.plist
echo "scheduled the super-agent: tick 2 min, metrics 30 min, watchdog 15 min"
