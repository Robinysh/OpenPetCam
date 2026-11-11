#!/usr/bin/env bash
# Runs on the server as root (deploy.sh syncs server/, models/ and site.env next to it, then calls it).
# Idempotent: sets up a fresh Ubuntu 24.04 box, and is the only way config reaches an existing one.
# Never overwrites secrets or keys that already exist.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
[ -f "$SRC/site.env" ] || { echo "missing $SRC/site.env (copy config/site.env.example to config/site.env and fill it in)"; exit 1; }
set -a; . "$SRC/site.env"; set +a
: "${PET_NAME:?set PET_NAME in config/site.env}" "${PUBLIC_HOST:?set PUBLIC_HOST}" "${PUBLIC_IP:?set PUBLIC_IP}"
CAMERA=${CAMERA:-thingino}   # thingino: a Thingino camera dialling in over WireGuard (the reference). rpi: a Pi camera on this machine.
CAMERA_WG_IP=${CAMERA_WG_IP:-10.77.0.2}; SERVER_WG_IP=${SERVER_WG_IP:-10.77.0.1}; TZ=${TZ:-UTC}
MEDIAMTX_VERSION=v1.21.3
declare -A MEDIAMTX_SHA256=(
  [amd64]=0a2ea7eba96023631efa1bea7804823a9145b043b6646a8a05798d4ac24ed635
  [arm64]=27f038237cb5f77a724f6afc58eca2798f2bc4edf9989895631a80eeb41a4fe2
)
render() {  # fill @PLACEHOLDERS@ from site.env
  sed -e "s|@PUBLIC_HOST@|$PUBLIC_HOST|g" -e "s|@PUBLIC_IP@|$PUBLIC_IP|g" \
      -e "s|@CAMERA_WG_IP@|$CAMERA_WG_IP|g" -e "s|@SERVER_WG_IP@|$SERVER_WG_IP|g" "$1"
}

# Packages (skipped when present). Ubuntu's own caddy package is new enough (forward_auth).
need=()
pkgs="caddy ffmpeg python3-venv python3-numpy python3-pil curl"
[ "$CAMERA" = rpi ] || pkgs="$pkgs wireguard-tools"
for p in $pkgs; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then
  DEBIAN_FRONTEND=noninteractive apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${need[@]}"
fi

# MediaMTX, pinned and checksum-verified.
if [ "$(/opt/mediamtx/mediamtx --version 2>/dev/null)" != "$MEDIAMTX_VERSION" ]; then
  arch=$(dpkg --print-architecture)
  tgz=$(mktemp)
  curl -fsSL -m 120 -o "$tgz" "https://github.com/bluenviron/mediamtx/releases/download/$MEDIAMTX_VERSION/mediamtx_${MEDIAMTX_VERSION}_linux_${arch}.tar.gz"
  echo "${MEDIAMTX_SHA256[$arch]}  $tgz" | sha256sum -c --quiet -
  install -d -m 755 /opt/mediamtx && tar -xzf "$tgz" -C /opt/mediamtx mediamtx && rm -f "$tgz"
fi

id mediamtx >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin mediamtx
id petcam   >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin petcam
usermod -aG mediamtx petcam   # lets the alert checker see recording freshness (read-only)

if [ "$CAMERA" != rpi ]; then
# WireGuard: the camera dials out to this server, so no port is opened at home.
if [ ! -f /etc/wireguard/wg0.conf ]; then
  ( umask 077
    [ -f /etc/wireguard/server.key ] || wg genkey > /etc/wireguard/server.key
    {
      echo "[Interface]"; echo "Address = $SERVER_WG_IP/24"; echo "ListenPort = 51820"
      echo "PostUp = wg set %i private-key /etc/wireguard/server.key"
      if [ -n "${CAMERA_WG_PUBKEY:-}" ]; then
        echo; echo "[Peer]"; echo "# camera"; echo "PublicKey = $CAMERA_WG_PUBKEY"; echo "AllowedIPs = $CAMERA_WG_IP/32"
      fi
    } > /etc/wireguard/wg0.conf )
  systemctl enable --now wg-quick@wg0
elif [ -n "${CAMERA_WG_PUBKEY:-}" ] && ! grep -qF "$CAMERA_WG_PUBKEY" /etc/wireguard/wg0.conf; then
  printf '\n[Peer]\n# camera\nPublicKey = %s\nAllowedIPs = %s/32\n' "$CAMERA_WG_PUBKEY" "$CAMERA_WG_IP" >> /etc/wireguard/wg0.conf
  systemctl restart wg-quick@wg0
fi
echo "WireGuard server public key (for the camera): $(wg pubkey < /etc/wireguard/server.key 2>/dev/null || echo '?')"
fi

if [ "$CAMERA" != rpi ]; then   # cloud-VM housekeeping; a Pi keeps its own services
# Slim the box: a 512 MB / 5%-CPU-baseline VM can't carry desktop/firmware services. fwupd alone
# took 80 MB and pushed the box into swap + CPU throttling (2026-11-01). Security updates
# (unattended-upgrades) stay on. SSH uses our own key, so snapd / the SSM agent are not needed.
for unit in fwupd.service fwupd-refresh.timer multipathd.service multipathd.socket ModemManager.service \
            udisks2.service snapd.service snapd.socket snapd.seeded.service snapd.snap-repair.timer \
            snap.amazon-ssm-agent.amazon-ssm-agent.service motd-news.timer man-db.timer \
            update-notifier-download.timer update-notifier-motd.timer; do
  systemctl disable --now "$unit" >/dev/null 2>&1 || true
done
systemctl mask fwupd.service fwupd-refresh.service >/dev/null 2>&1 || true
fi

install -d -m 755 /etc/mediamtx /opt/petcam
install -d -m 750 -o root -g petcam /etc/petcam
grep -vE '^(SSH_|#|$)' "$SRC/site.env" > /etc/petcam/site.env   # laptop-only SSH settings stay on the laptop
chown root:petcam /etc/petcam/site.env && chmod 640 /etc/petcam/site.env
install -d -m 750 -o mediamtx -g mediamtx /var/lib/mediamtx /var/lib/mediamtx/recordings

# Python env for the watcher: YOLOX (onnxruntime) + YAMNet (ai-edge-litert; tflite-runtime has no
# Python 3.12 wheel). Models are copied from ../models when present; YAMNet's checksum is pinned.
[ -x /opt/petcam/venv/bin/python ] || python3 -m venv --system-site-packages /opt/petcam/venv
/opt/petcam/venv/bin/python -c "import onnxruntime, ai_edge_litert" 2>/dev/null || \
  /opt/petcam/venv/bin/pip install -q onnxruntime ai-edge-litert
/opt/petcam/venv/bin/python -c "import pywebpush" 2>/dev/null || /opt/petcam/venv/bin/pip install -q pywebpush
install -d -m 755 /opt/petcam/models
for m in yolox_nano.onnx yolox_nano_dyn.onnx yamnet.tflite yamnet_class_map.csv mobilenetv2-12.onnx petnet.onnx verifier.json follow_policy.json follow_ppo.onnx room_map.json; do
  [ -f "$SRC/models/$m" ] && install -m 644 "$SRC/models/$m" /opt/petcam/models/
done
# A learned Follow policy that is no longer shipped must stop running: set it aside (not deleted, for the record).
# Only when this source ships models at all (a ./deploy.sh from a checkout that has them): a GitHub release carries
# no models, and must not retire the policy this camera learned (review 2026-11-04).
for m in follow_policy.json follow_ppo.onnx room_map.json; do
  if [ -f "$SRC/models/yolox_nano.onnx" ] && [ ! -f "$SRC/models/$m" ] && [ -f "/opt/petcam/models/$m" ]; then
    mv "/opt/petcam/models/$m" "/opt/petcam/models/$m.retired-$(date +%Y%m%d%H%M%S)"
  fi
done
echo "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de  /opt/petcam/models/yamnet.tflite" | sha256sum -c --quiet -
# Your room bundle (sim/world/app_bundle.py): the inside of your home, readable by the app only. Replaced as a whole,
# so a removed texture does not linger; set aside (not deleted) when a checkout with models no longer ships one.
if [ -f "$SRC/models/room/room.json" ]; then
  rm -rf /opt/petcam/models/room.new && install -d -m 750 -g petcam /opt/petcam/models/room.new
  for f in "$SRC"/models/room/*; do install -m 640 -g petcam "$f" /opt/petcam/models/room.new/; done
  [ -d /opt/petcam/models/room ] && mv /opt/petcam/models/room "/opt/petcam/models/room.old-$$"
  mv /opt/petcam/models/room.new /opt/petcam/models/room && rm -rf "/opt/petcam/models/room.old-$$"
elif [ -f "$SRC/models/yolox_nano.onnx" ] && [ -d /opt/petcam/models/room ]; then
  mv /opt/petcam/models/room "/opt/petcam/models/room.retired-$(date +%Y%m%d%H%M%S)"
fi

render $SRC/mediamtx.yml > /etc/mediamtx/mediamtx.yml && chmod 644 /etc/mediamtx/mediamtx.yml
install -m 644 $SRC/petconf.py $SRC/agentdb.py $SRC/goalmonitor.py $SRC/startupmetrics.py $SRC/reportstate.py $SRC/requestboard.py $SRC/requests.js $SRC/requests.css $SRC/clipmedia.py $SRC/followmetrics.py $SRC/viewstats.py $SRC/tracker.py $SRC/roommap.py $SRC/soundfind.py $SRC/roomcam.py $SRC/roomapi.py $SRC/roomfollow.py $SRC/fasttrack.py $SRC/scheduled_sweep.py $SRC/room.js $SRC/follow_obs.py $SRC/ppo_follow.py $SRC/app.py $SRC/watch.py $SRC/detector.py $SRC/alert.py $SRC/update.py $SRC/index.html $SRC/agent.html $SRC/login.html $SRC/join.html \
  $SRC/manifest.webmanifest $SRC/icon.png $SRC/apple-touch-icon.png $SRC/favicon.png $SRC/favicon.svg $SRC/icon-maskable.png $SRC/badge.png $SRC/logo.png $SRC/logo-light.png $SRC/themes.css $SRC/theme-controller.js $SRC/theme-data.js $SRC/theme-app.js $SRC/sw.js /opt/petcam/
install -d -m 755 /opt/petcam/vendor && install -m 644 $SRC/vendor/three.module.min.js $SRC/vendor/OrbitControls.js /opt/petcam/vendor/
render $SRC/Caddyfile > /etc/caddy/Caddyfile && chmod 644 /etc/caddy/Caddyfile
if [[ "$PUBLIC_HOST" == *.ts.net ]]; then
  # Inside the tailnet MagicDNS resolves the machine's short name: http://<name> lands on the real HTTPS address.
  printf '\nhttp://%s {\n\tredir https://%s{uri}\n}\n' "${PUBLIC_HOST%%.*}" "$PUBLIC_HOST" >> /etc/caddy/Caddyfile
fi
install -m 750 -o root -g root $SRC/set-secrets.sh /opt/petcam/set-secrets.sh

# Camera stream source.
if [ "$CAMERA" = rpi ]; then
  # A Pi camera on this machine: MediaMTX reads it itself (rpiCamera). Same path names as the Thingino setup
  # (cam = recorded main stream, sub = 640x360 MJPEG for the watcher), so everything downstream is unchanged.
  # No secrets in this file; it is rewritten on every install from config/site.env.
  RPI_W=${RPI_CAM_WIDTH:-1280}; RPI_H=${RPI_CAM_HEIGHT:-720}; RPI_FPS=${RPI_CAM_FPS:-20}
  {
    echo "MTX_PATHS_CAM_SOURCE=rpiCamera"
    echo "MTX_PATHS_CAM_RPICAMERAWIDTH=$RPI_W"
    echo "MTX_PATHS_CAM_RPICAMERAHEIGHT=$RPI_H"
    echo "MTX_PATHS_CAM_RPICAMERAFPS=$RPI_FPS"
    echo "MTX_PATHS_CAM_RPICAMERABITRATE=${RPI_CAM_BITRATE:-1500000}"
    echo "MTX_PATHS_CAM_RPICAMERAIDRPERIOD=$((RPI_FPS * 2))"
    echo "MTX_PATHS_CAM_RPICAMERATEXTOVERLAYENABLE=yes"     # a ticking clock in the picture: the freeze watchdog's signal
    echo "MTX_PATHS_CAM_RPICAMERAHFLIP=${RPI_CAM_HFLIP:-no}"
    echo "MTX_PATHS_CAM_RPICAMERAVFLIP=${RPI_CAM_VFLIP:-no}"
    [ -z "${RPI_CAM_MODE-1640:1232:10:P}" ] || echo "MTX_PATHS_CAM_RPICAMERAMODE=${RPI_CAM_MODE-1640:1232:10:P}"
    echo "MTX_PATHS_SUB_SOURCE=rpiCamera"
    echo "MTX_PATHS_SUB_RPICAMERASECONDARY=yes"
    echo "MTX_PATHS_SUB_RPICAMERAWIDTH=640"
    echo "MTX_PATHS_SUB_RPICAMERAHEIGHT=360"
    echo "MTX_PATHS_SUB_RPICAMERAFPS=5"
    echo "MTX_PATHS_SUB_SOURCEONDEMAND=no"                  # a secondary stream lives and dies with the main one
    echo "MTX_API=yes"                                      # localhost only: the app's privacy pause releases the camera through it
    echo "MTX_APIADDRESS=127.0.0.1:9997"
  } > /etc/petcam/camera.env
# Default Thingino RTSP login until it is changed on the camera.
elif [ ! -f /etc/petcam/camera.env ]; then
  echo "MTX_PATHS_CAM_SOURCE=rtsp://thingino:thingino@$CAMERA_WG_IP:554/ch0" > /etc/petcam/camera.env
fi
# The substream path for petcam-watch: same login, /ch1 (added to installs made before it existed).
if [ "$CAMERA" != rpi ] && ! grep -q '^MTX_PATHS_SUB_SOURCE=' /etc/petcam/camera.env; then
  sed -n 's#^MTX_PATHS_CAM_SOURCE=\(.*\)/ch0$#MTX_PATHS_SUB_SOURCE=\1/ch1#p' /etc/petcam/camera.env >> /etc/petcam/camera.env
fi
chown root:mediamtx /etc/petcam/camera.env && chmod 640 /etc/petcam/camera.env
# mediamtx must traverse /etc/petcam to read its env file (systemd reads it as root, so this is belt and braces).

# App secrets: session signing key now; password hash and camera web login are added by set-secrets.sh.
if [ ! -f /etc/petcam/app.env ]; then
  printf 'APP_SECRET=%s\nCAM_WEB_USER=root\n' "$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" > /etc/petcam/app.env
fi
chown root:petcam /etc/petcam/app.env && chmod 640 /etc/petcam/app.env

# SSH key for the fast camera channel (app.py CamShell); the app authorises it on the camera itself.
if [ "$CAMERA" != rpi ]; then
  install -d -m 700 -o petcam -g petcam /var/lib/petcam/ssh   # only this folder: Caddy serves media from /var/lib/petcam
  [ -f /var/lib/petcam/ssh/id_ed25519 ] || sudo -u petcam ssh-keygen -q -t ed25519 -N '' -C petcam-server -f /var/lib/petcam/ssh/id_ed25519
fi

# Web push (VAPID) keys, generated once on the server.
/opt/petcam/venv/bin/python $SRC/gen_vapid.py /etc/petcam
chown root:petcam /etc/petcam/vapid_private.pem /etc/petcam/vapid_public.txt
chmod 640 /etc/petcam/vapid_private.pem; chmod 644 /etc/petcam/vapid_public.txt

MTX_GROUPS=""
[ "$CAMERA" != rpi ] || MTX_GROUPS="SupplementaryGroups=video render"   # the camera and its buffers: /dev/video*, /dev/media*, /dev/dma_heap
cat > /etc/systemd/system/mediamtx.service <<UNIT
[Unit]
Description=MediaMTX (petcam video)
After=network-online.target wg-quick@wg0.service
Wants=network-online.target

[Service]
User=mediamtx
Group=mediamtx
EnvironmentFile=/etc/petcam/camera.env
ExecStart=/opt/mediamtx/mediamtx /etc/mediamtx/mediamtx.yml
Restart=always
RestartSec=5
TimeoutStopSec=10
NoNewPrivileges=yes
ProtectSystem=strict
$MTX_GROUPS
ReadWritePaths=/var/lib/mediamtx
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/petcam-app.service <<UNIT
[Unit]
Description=petcam web app
After=network-online.target

[Service]
User=petcam
Group=petcam
StateDirectory=petcam
Environment=TZ=$TZ
ExecStart=/usr/bin/python3 /opt/petcam/app.py
Restart=always
RestartSec=3
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/petcam-watch.service <<UNIT
[Unit]
Description=petcam watcher (pet detection, freeze watchdog, sounds, clips, timelapse)
After=network-online.target mediamtx.service wg-quick@wg0.service

[Service]
User=petcam
Group=petcam
StateDirectory=petcam
Environment=TZ=$TZ
ExecStart=/opt/petcam/venv/bin/python /opt/petcam/watch.py
Restart=always
RestartSec=5
Nice=5
CPUQuota=60%
MemoryMax=300M
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT

# Absence alerter: separate from the watcher so a dead watcher is still reported.
cat > /etc/systemd/system/petcam-alert.service <<UNIT
[Unit]
Description=petcam alerts (watcher/video absence, camera problems, barking) via web push
After=network-online.target

[Service]
Type=oneshot
User=petcam
Group=petcam
Environment=TZ=$TZ
ExecStart=/usr/bin/timeout -k 10 80 /usr/bin/python3 /opt/petcam/alert.py
TimeoutStartSec=100
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/petcam
ProtectHome=yes
PrivateTmp=yes
UNIT
cat > /etc/systemd/system/petcam-alert.timer <<UNIT
[Unit]
Description=Run petcam alerts every 3 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=3min
AccuracySec=15s

[Install]
WantedBy=timers.target
UNIT

# Scheduled sweeps (server/scheduled_sweep.py): `scheduled_sweep.py schedule TAG "YYYY-MM-DD HH:MM" poses.json` writes a
# one-shot timer for this unit; the run has a hard wall-clock limit, and alert.py says when one fails or never ran.
cat > /etc/systemd/system/petcam-sweep@.service <<UNIT
[Unit]
Description=petcam scheduled sweep %i
After=network-online.target petcam-app.service

[Service]
Type=oneshot
Environment=TZ=$TZ
ExecStart=/usr/bin/timeout -k 15 1500 /opt/petcam/venv/bin/python /opt/petcam/scheduled_sweep.py run %i
TimeoutStartSec=1560
UNIT

# Updates from GitHub releases (server/update.py): daily at a quiet hour, as root because it runs this script.
cat > /etc/systemd/system/petcam-update.service <<UNIT
[Unit]
Description=petcam: install a newer OpenPetCam release if there is one
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
Environment=TZ=$TZ
ExecStart=/usr/bin/timeout -k 30 2400 /usr/bin/python3 /opt/petcam/update.py
TimeoutStartSec=2500
Nice=10
UNIT
cat > /etc/systemd/system/petcam-update.timer <<UNIT
[Unit]
Description=Check for OpenPetCam releases once a day

[Timer]
OnCalendar=*-*-* 04:20 $TZ
RandomizedDelaySec=40min
Persistent=true

[Install]
WantedBy=timers.target
UNIT

# The version this is, and a copy of the source (no settings, no models) so update.py can roll back exactly.
if [ -f "$SRC/VERSION" ]; then install -m 644 "$SRC/VERSION" /opt/petcam/VERSION; fi
if [ "$SRC" != /opt/petcam/source ]; then
  rm -rf /opt/petcam/source.new && mkdir -p /opt/petcam/source.new
  ( cd "$SRC" && tar --exclude=./site.env --exclude=./models --exclude=__pycache__ -cf - . ) | tar -xf - -C /opt/petcam/source.new
  rm -rf /opt/petcam/source && mv /opt/petcam/source.new /opt/petcam/source && chmod -R go-w /opt/petcam/source
fi

if [ "$CAMERA" = rpi ]; then
  # The services can't restart MediaMTX themselves (NoNewPrivileges, not root): the watcher touches this file and a
  # root path unit does it. This is what the freeze watchdog and the self-heal use instead of a camera CGI.
  install -d -m 755 -o petcam -g petcam /var/lib/petcam
  touch /var/lib/petcam/restart_camera && chown petcam:petcam /var/lib/petcam/restart_camera
  cat > /etc/systemd/system/petcam-camrestart.path <<UNIT
[Unit]
Description=Restart the camera stream when petcam-watch asks

[Path]
PathChanged=/var/lib/petcam/restart_camera

[Install]
WantedBy=multi-user.target
UNIT
  cat > /etc/systemd/system/petcam-camrestart.service <<UNIT
[Unit]
Description=Restart MediaMTX (camera stream)

[Service]
Type=oneshot
ExecStart=/bin/systemctl restart mediamtx
UNIT
  # HTTPS over Tailscale: Caddy fetches the *.ts.net certificate from tailscaled, which must allow the caddy user.
  if [[ "$PUBLIC_HOST" == *.ts.net ]] && command -v tailscale >/dev/null 2>&1; then
    if ! grep -qs '^TS_PERMIT_CERT_UID=caddy' /etc/default/tailscaled; then
      sed -i '/^TS_PERMIT_CERT_UID=/d' /etc/default/tailscaled 2>/dev/null || true
      echo 'TS_PERMIT_CERT_UID=caddy' >> /etc/default/tailscaled
      systemctl restart tailscaled
      sleep 3
    fi
  fi
fi

systemctl daemon-reload
[ "$CAMERA" != rpi ] || systemctl enable --now petcam-camrestart.path >/dev/null 2>&1
systemctl enable --now petcam-alert.timer >/dev/null 2>&1
systemctl enable --now petcam-update.timer >/dev/null 2>&1
systemctl enable --now mediamtx petcam-app petcam-watch >/dev/null 2>&1
systemctl restart mediamtx petcam-app petcam-watch
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 || caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl reload caddy || systemctl restart caddy
sleep 3
systemctl is-active mediamtx petcam-app caddy

# Ubuntu's firmware updater has nothing to update on a cloud VM; its timer fails and shows as a failed unit in QA.
if [ "$CAMERA" != rpi ]; then
  systemctl disable --now fwupd-refresh.timer >/dev/null 2>&1 || true
  systemctl reset-failed fwupd-refresh.service >/dev/null 2>&1 || true
fi
