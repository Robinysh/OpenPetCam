#!/usr/bin/env bash
# One-shot health report for the petcam server (read-only).
set -u
set -a; . "$(dirname "$0")/../config/site.env"; set +a
KEY=${SSH_KEY/#\~/$HOME}; HOST=$SSH_HOST
echo "== $(date '+%H:%M:%S') local"
if [ "$HOST" = local ]; then RUN=(bash -s); else RUN=(ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=20 $HOST 'bash -s'); fi
"${RUN[@]}" <<'R'
systemctl is-active caddy mediamtx petcam-app petcam-watch petcam-alert.timer $(systemctl cat wg-quick@wg0 >/dev/null 2>&1 && echo wg-quick@wg0) | paste -s | sed "s/^/services: /"
sudo python3 - <<PY
import json, time, sqlite3
d = json.load(open("/var/lib/petcam/live.json"))
print("watcher: %s, age %.0fs, pet present=%s, audio_ok=%s, imaging=%s" % (d["status"], time.time() - d["t"], d["present"], d.get("audio_ok"), (d.get("imaging") or {}).get("daynight_mode")))
con = sqlite3.connect("/var/lib/petcam/petcam.db")
print("obs last 10 min:", con.execute("select count(*), round(max(score),2), sum(present) from obs where ts > ?", (time.time() - 600,)).fetchone())
print("events last 2 h:", con.execute("select kind, count(*) from events where ts > ? group by kind", (time.time() - 7200,)).fetchall())
PY
free -m | awk "/Mem/{print \"mem: used \" \$3 \" MB, available \" \$7 \" MB\"} /Swap/{print \"swap used: \" \$3 \" MB\"}"
top -b -d 20 -n 2 | awk "/^top/{n++} n==2" | sed -n 3p
if systemctl cat wg-quick@wg0 >/dev/null 2>&1; then sudo wg show wg0 latest-handshakes | awk "{print \"wireguard handshake age: \" systime()-\$2 \"s\"}"; fi
df -h / | awk "NR==2{print \"disk: \" \$3 \" used, \" \$4 \" free\"}"
R
if [ -n "${LIGHTSAIL_INSTANCE:-}" ]; then   # optional: AWS Lightsail CPU burst credits
  timeout -k 2 30 aws lightsail get-instance-metric-data --region "${LIGHTSAIL_REGION:-us-east-1}" --instance-name "$LIGHTSAIL_INSTANCE" \
    --metric-name BurstCapacityPercentage --period 300 --start-time "$(date -u -v-15M +%Y-%m-%dT%H:%M:%SZ)" \
    --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --unit Percent --statistics Average --query 'metricData[].average' \
    --output text | awk '{print "burst capacity %: " $0}'
fi
