#!/bin/sh
# Verified software capture stop. This does not electrically disconnect the camera.
# Installation preserves the reviewed service and does not change its running state.
set -eu
OWNER=openpetcam-privacy-guard-v1
SERVICE_SHA=5ec4c4315f02b68a3a0ee01b539fb3fc1af54d7dabd875a58709da4e06e53907
STOCK_SHA=3f23b94eab5463db62c069e83dd48cc535ce2d91b6990f216cb0efb587f9d25d
FIXED_SHA=45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b
SDK_SHA=bd904ec8f4123965ded8c0c57c25ad5c3ae0fdc17aa4a6374278a7c822cb31d9
R=${OPC_PRIVACY_TEST_ROOT:-}
if [ -n "$R" ]; then
  case "$R" in /*) ;; *) echo 'Test root must be absolute.' >&2; exit 2;; esac
  [ -f "$R/.opc-privacy-test-root" ] || { echo 'Refusing unmarked test root.' >&2; exit 2; }
fi
PKG=$R/etc/opc-privacy
SERVICE=$R/etc/init.d/S31prudynt
ROM_SERVICE=$R/rom/etc/init.d/S31prudynt
ROM_BINARY=$R/rom/usr/bin/prudynt
BINARY=$R/usr/bin/prudynt
SDK=$R/usr/lib/libimp.so
PAUSED=$PKG/paused
RUN=$R/tmp/opc-privacy
LOCK=$R/tmp/opc-privacy.lock
LOCKED=0
STAGE=
SERVICE_TEMP=
fail() { echo "opc-privacy: $*" >&2; exit 1; }
hash() { sha256sum "$1" | awk '{print $1}'; }
cleanup() {
  set +e
  [ -z "$STAGE" ] || rm -rf "$STAGE"
  [ -z "$SERVICE_TEMP" ] || rm -f "$SERVICE_TEMP"
  # Minimal camera BusyBox has no rmdir. Remove only the lock this process created.
  if [ "$LOCKED" = 1 ]; then rm -rf "$LOCK"; fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
lock() {
  mkdir "$LOCK" 2>/dev/null || fail 'Another privacy operation is running; try again.'
  LOCKED=1
}
need_tools() {
  for tool in sha256sum awk df cp mv chmod mkdir rm cat pidof sleep sync mktemp wc; do
    command -v "$tool" >/dev/null 2>&1 || fail "Required command unavailable: $tool"
  done
}
owned() { [ -f "$PKG/owner" ] && [ "$(cat "$PKG/owner")" = "$OWNER" ]; }
paused() { [ -e "$PAUSED" ] || [ -L "$PAUSED" ]; }
running() { pidof prudynt >/dev/null 2>&1; }
check_target() {
  [ -f "$ROM_SERVICE" ] && [ "$(hash "$ROM_SERVICE")" = "$SERVICE_SHA" ] || fail 'Immutable service differs; no installation changes made.'
  [ -f "$ROM_BINARY" ] && [ "$(hash "$ROM_BINARY")" = "$STOCK_SHA" ] || fail 'Immutable streamer differs; no installation changes made.'
  [ -f "$SDK" ] && [ "$(hash "$SDK")" = "$SDK_SHA" ] || fail 'Target SDK differs; no installation changes made.'
  current=$(hash "$BINARY")
  [ "$current" = "$STOCK_SHA" ] || [ "$current" = "$FIXED_SHA" ] || fail 'Active streamer differs from the reviewed images.'
}
check_package() {
  owned || fail 'No owned privacy guard is installed.'
  [ -f "$PKG/service.original" ] && [ ! -L "$PKG/service.original" ] && [ "$(hash "$PKG/service.original")" = "$SERVICE_SHA" ] || fail 'Original service identity differs.'
  [ -f "$PKG/control" ] && [ ! -L "$PKG/control" ] && [ "$(hash "$PKG/control")" = "$(cat "$PKG/control.sha256")" ] || fail 'Control identity differs.'
  [ -f "$PKG/service.gate" ] && [ ! -L "$PKG/service.gate" ] && [ "$(hash "$PKG/service.gate")" = "$(cat "$PKG/gate.sha256")" ] || fail 'Stored gate identity differs.'
}
check_gate() {
  [ -f "$SERVICE" ] && [ ! -L "$SERVICE" ] && [ "$(hash "$SERVICE")" = "$(cat "$PKG/gate.sha256")" ] || fail 'Active service gate identity differs.'
}
runtime() {
  if [ -e "$RUN" ] || [ -L "$RUN" ]; then
    [ -d "$RUN" ] && [ ! -L "$RUN" ] && [ -f "$RUN/owner" ] && [ "$(cat "$RUN/owner")" = "$OWNER" ] || fail 'Unrelated runtime path exists.'
  else
    mkdir -m 700 "$RUN"
    printf '%s\n' "$OWNER" > "$RUN/owner"
  fi
}
mark_paused() {
  # Persist before any stop, so a simultaneous reboot/start cannot reopen capture.
  printf '1\n' > "$PKG/paused.new.$$"
  chmod 600 "$PKG/paused.new.$$"
  mv "$PKG/paused.new.$$" "$PAUSED"
  sync
}
stop_capture() {
  runtime
  "$PKG/service.original" stop > "$RUN/service.log" 2>&1 || true
  attempts=0
  while running; do
    attempts=$((attempts + 1))
    [ "$attempts" -lt 20 ] || fail 'Capture process did not exit. Pause remains requested; shutdown is not verified.'
    sleep 1
  done
  # The ISP may still be completing teardown after the PID disappears.
  sleep 2
}
start_capture() {
  check_target
  runtime
  # Never launch a duplicate against an already running encoder/ISP.
  running && return 0
  "$PKG/service.original" start > "$RUN/service.log" 2>&1 || return 1
  attempts=0
  until running; do
    attempts=$((attempts + 1))
    [ "$attempts" -lt 6 ] || return 1
    sleep 1
  done
}
print_status() {
  is_paused=false
  is_running=false
  verified=false
  mode=capture_unavailable
  if paused; then is_paused=true; fi
  if running; then is_running=true; fi
  if [ "$is_paused" = true ] && [ "$is_running" = false ]; then
    verified=true
    mode=capture_stopped
  elif [ "$is_paused" = false ] && [ "$is_running" = true ]; then
    verified=true
    mode=capture_running
  fi
  printf '{"mode":"%s","privacy":%s,"paused":%s,"running":%s,"verified":%s,"boot_persistent":true}\n' "$mode" "$is_paused" "$is_paused" "$is_running" "$verified"
}
install_gate() {
  SERVICE_TEMP=$R/etc/init.d/.opc-privacy-service.$$
  cp "$PKG/service.gate" "$SERVICE_TEMP"
  chmod 755 "$SERVICE_TEMP"
  mv "$SERVICE_TEMP" "$SERVICE"
  SERVICE_TEMP=
}
usage() {
  cat <<'USAGE'
Usage:
  install_privacy_guard.sh dry-run
  install_privacy_guard.sh install
  /etc/opc-privacy/control pause
  /etc/opc-privacy/control resume
  /etc/opc-privacy/control status
  /etc/opc-privacy/control uninstall
Installation changes no running process. Pause stops capture and persists across boot.
Resume explicitly reopens capture. Status verifies software/process state, not electrical isolation.
Uninstall is refused while paused; resume explicitly before removing the guard.
USAGE
}
command=${1:-}
[ -n "$command" ] || { usage; exit 2; }
shift
case "$command" in
  dry-run|install)
    [ "$#" -eq 0 ] || fail 'Unexpected installation argument.'
    need_tools
    check_target
    lock
    if [ -e "$PKG" ] || [ -L "$PKG" ]; then
      check_package
      [ "$(hash "$0")" = "$(cat "$PKG/control.sha256")" ] || fail 'Installed guard differs from the incoming version; refusing an implicit upgrade.'
      if [ "$(hash "$SERVICE")" != "$(cat "$PKG/gate.sha256")" ]; then
        [ "$(hash "$SERVICE")" = "$SERVICE_SHA" ] || fail 'Unrelated active service exists.'
        [ "$command" = dry-run ] || install_gate
      fi
      echo 'opc-privacy: Identical owned guard already installed; running state unchanged.'
      exit 0
    fi
    [ -f "$SERVICE" ] && [ ! -L "$SERVICE" ] && [ "$(hash "$SERVICE")" = "$SERVICE_SHA" ] || fail 'Active service differs; refusing to replace it.'
    available=$(df -Pk "$R/etc" | awk 'NR==2 {print $4}')
    script_bytes=$(wc -c < "$0")
    required=$((128 + (script_bytes * 2 + 8191) / 1024))
    [ "$available" -ge "$required" ] || fail 'Insufficient overlay reserve for owned service backup and guard.'
    if [ "$command" = dry-run ]; then
      echo 'opc-privacy: Preflight passed; original service and running state unchanged.'
      exit 0
    fi
    STAGE=$(mktemp -d "$R/etc/.opc-privacy-stage.XXXXXX")
    chmod 700 "$STAGE"
    cp "$SERVICE" "$STAGE/service.original"
    chmod 755 "$STAGE/service.original"
    [ "$(hash "$STAGE/service.original")" = "$SERVICE_SHA" ] || fail 'Service changed while copying.'
    cp "$0" "$STAGE/control"
    chmod 755 "$STAGE/control"
    hash "$STAGE/control" > "$STAGE/control.sha256"
    printf '%s\n' "$OWNER" > "$STAGE/owner"
    printf '%s\n' '#!/bin/sh' '# OpenPetCam owned privacy gate: missing/corrupt control fails closed.' "exec \"$PKG/control\" service \"\$@\"" > "$STAGE/service.gate"
    chmod 755 "$STAGE/service.gate"
    hash "$STAGE/service.gate" > "$STAGE/gate.sha256"
    mv "$STAGE" "$PKG"
    STAGE=
    install_gate
    sync
    echo 'opc-privacy: Owned privacy guard installed; running capture unchanged.'
    ;;
  status)
    [ "$#" -eq 0 ] || fail 'Unexpected status argument.'
    need_tools; check_package; check_gate
    print_status
    ;;
  pause)
    [ "$#" -eq 0 ] || fail 'Unexpected pause argument.'
    need_tools; lock; check_package; check_gate
    mark_paused
    stop_capture
    print_status
    ;;
  resume)
    [ "$#" -eq 0 ] || fail 'Unexpected resume argument.'
    need_tools; lock; check_package; check_gate
    if paused && running; then stop_capture; fi
    # Keep the durable pause until explicit startup succeeds. Fatal preconditions
    # (changed SDK, unavailable runtime path) must not clear it before returning.
    if ! start_capture; then
      mark_paused
      stop_capture
      fail 'Capture could not start; persistent pause restored.'
    fi
    rm -f "$PAUSED"
    if ! sync; then
      mark_paused
      stop_capture
      fail 'Resume state could not be synced; persistent pause restored.'
    fi
    print_status
    ;;
  service)
    [ "$#" -eq 1 ] || fail 'One service action is required.'
    action=$1
    case "$action" in start|stop|restart) ;; *) fail 'Unsupported service action.';; esac
    need_tools; lock; check_package; check_gate
    case "$action" in
      stop) stop_capture;;
      start)
        if paused; then
          # Also stop an unexpected existing process rather than merely skip a new start.
          if running; then stop_capture; fi
        else start_capture || fail 'Capture could not start.'; fi
        ;;
      restart)
        stop_capture
        if ! paused; then start_capture || fail 'Capture could not restart.'; fi
        ;;
    esac
    print_status
    ;;
  uninstall)
    [ "$#" -eq 0 ] || fail 'Unexpected uninstall argument.'
    need_tools; lock; check_package; check_gate
    paused && fail 'Privacy is paused. Explicitly resume before removing the guard.'
    SERVICE_TEMP=$R/etc/init.d/.opc-privacy-service.$$
    cp "$PKG/service.original" "$SERVICE_TEMP"
    chmod 755 "$SERVICE_TEMP"
    [ "$(hash "$SERVICE_TEMP")" = "$SERVICE_SHA" ] || fail 'Original service restore failed.'
    mv "$SERVICE_TEMP" "$SERVICE"
    SERVICE_TEMP=
    rm -rf "$PKG"
    if [ -f "$RUN/owner" ] && [ "$(cat "$RUN/owner")" = "$OWNER" ]; then rm -rf "$RUN"; fi
    sync
    echo 'opc-privacy: Original service restored; running capture unchanged.'
    ;;
  *) usage; exit 2;;
esac
