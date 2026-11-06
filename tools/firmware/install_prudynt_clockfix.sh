#!/bin/sh
# OpenPetCam verified Prudynt clock repair: compressed overlay + RAM bind mount.
# No action is taken without an explicit command. ROM and camera settings are never overwritten.
set -eu
STOCK_SHA=3f23b94eab5463db62c069e83dd48cc535ce2d91b6990f216cb0efb587f9d25d
SDK_SHA=bd904ec8f4123965ded8c0c57c25ad5c3ae0fdc17aa4a6374278a7c822cb31d9
FIXED_SHA=45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b
FIXED_BYTES=967000
OWNER=openpetcam-prudynt-clockfix-v1
GUARD_SECONDS=180
RESERVE_KIB=128
R=${OPC_CLOCKFIX_TEST_ROOT:-}
if [ -n "$R" ]; then
  [ -f "$R/.opc-clockfix-test-root" ] || { echo 'Refusing unmarked test root.' >&2; exit 2; }
  case "$R" in /*) ;; *) echo 'Test root must be absolute.' >&2; exit 2;; esac
fi
PKG=$R/etc/opc-clockfix
HOOK=$R/etc/init.d/S30opc-clockfix
SERVICE=$R/etc/init.d/S31prudynt
TARGET=$R/usr/bin/prudynt
ROM=$R/rom/usr/bin/prudynt
SDK=$R/usr/lib/libimp.so
RUN=$R/tmp/opc-clockfix
LOCK=$R/tmp/opc-clockfix.lock
MOUNTS=$R/proc/mounts
BOOT_ID=$R/proc/sys/kernel/random/boot_id
STAGE=
IMAGE_TMP=
HOOK_TMP=
BOOT_MOUNTED=0
BOOT_SAFE=0
NEW_GUARD_PID=
LOCKED=0
say() { echo "opc-clockfix: $*"; }
fail() { say "$*" >&2; exit 1; }
hash() { sha256sum "$1" | awk '{print $1}'; }
owned() { [ -f "$PKG/owner" ] && [ "$(cat "$PKG/owner")" = "$OWNER" ]; }
mounted() { awk -v target="$TARGET" '$2==target {found=1} END {exit !found}' "$MOUNTS"; }
free_kib() { df -Pk "$1" | awk 'NR==2 {print $4}'; }
cleanup() {
  set +e
  if [ "$BOOT_MOUNTED" = 1 ] && [ "$BOOT_SAFE" != 1 ]; then
    # A failed guard/state write must not leave an unguarded first-boot image.
    if [ -n "$NEW_GUARD_PID" ] && [ -r "$R/proc/$NEW_GUARD_PID/stat" ]; then
      parent=$(awk '{print $4}' "$R/proc/$NEW_GUARD_PID/stat")
      [ "$parent" != "$$" ] || kill "$NEW_GUARD_PID" 2>/dev/null
    fi
    if mounted && [ "$(hash "$TARGET")" = "$FIXED_SHA" ] && ! pidof prudynt >/dev/null 2>&1; then
      if ! umount "$TARGET"; then
        disable_hook
        say 'Boot setup failed and unmount failed; cold stock recovery required.' >&2
        sync; reboot
      fi
    fi
  fi
  [ -z "$STAGE" ] || rm -rf "$STAGE"
  [ -z "$IMAGE_TMP" ] || rm -f "$IMAGE_TMP"
  [ -z "$HOOK_TMP" ] || rm -f "$HOOK_TMP"
  if [ "$LOCKED" = 1 ]; then rm -rf "$LOCK"; fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
lock() { mkdir "$LOCK" 2>/dev/null || fail 'Another clock-fix operation is running.'; LOCKED=1; }
need_tools() {
  for tool in sha256sum gzip head wc awk df mount umount pidof sleep cp mv chmod mkdir rm tr cat mktemp sync; do
    command -v "$tool" >/dev/null 2>&1 || fail "Required command unavailable: $tool"
  done
}
check_target() {
  [ -f "$ROM" ] && [ "$(hash "$ROM")" = "$STOCK_SHA" ] || fail 'Immutable stock Prudynt hash differs; leaving the current installation alone.'
  [ -f "$SDK" ] && [ "$(hash "$SDK")" = "$SDK_SHA" ] || fail 'SDK libimp hash differs; leaving the current installation alone.'
  [ -x "$SERVICE" ] || fail 'Stock S31prudynt service is unavailable.'
}
check_package() {
  owned || fail 'No owned clock-fix package.'
  [ "$(cat "$PKG/sha256")" = "$FIXED_SHA" ] || fail 'Stored image identity differs.'
  [ "$(hash "$PKG/prudynt.gz")" = "$(cat "$PKG/gzip.sha256")" ] || fail 'Compressed image checksum failed.'
  [ "$(hash "$PKG/control")" = "$(cat "$PKG/control.sha256")" ] || fail 'Control script checksum failed.'
  [ "$(hash "$PKG/hook")" = "$(cat "$PKG/hook.sha256")" ] || fail 'Stored boot hook checksum failed.'
}
expand_checked() {
  archive=$1
  image=$2
  # Bound a corrupt/misidentified archive's RAM output before verifying the exact image.
  gzip -dc "$archive" | head -c "$((FIXED_BYTES + 1))" > "$image"
  [ "$(wc -c < "$image" | tr -d ' ')" = "$FIXED_BYTES" ] || fail 'Expanded image size differs.'
  [ "$(hash "$image")" = "$FIXED_SHA" ] || fail 'Expanded image checksum failed.'
  chmod 755 "$image"
}
write_atomic() {
  destination=$1
  content=$2
  printf '%s\n' "$content" > "$destination.new.$$"
  chmod 600 "$destination.new.$$"
  mv "$destination.new.$$" "$destination"
}
check_hook() {
  [ -f "$HOOK" ] && [ ! -L "$HOOK" ] && [ "$(hash "$HOOK")" = "$(cat "$PKG/hook.sha256")" ]
}
install_hook() {
  if [ -e "$HOOK" ] || [ -L "$HOOK" ]; then check_hook || fail 'An unrelated S30opc-clockfix hook exists.'; return; fi
  HOOK_TMP=$R/etc/init.d/.opc-clockfix-hook.$$
  cp "$PKG/hook" "$HOOK_TMP"
  chmod 755 "$HOOK_TMP"
  mv "$HOOK_TMP" "$HOOK"
  HOOK_TMP=
}
disable_hook() {
  if [ -e "$HOOK" ] || [ -L "$HOOK" ]; then
    check_hook || fail 'Boot hook identity changed; refusing to remove it.'
    mv "$HOOK" "$PKG/hook.disabled"
  fi
}
boot_identity() {
  # procfs reports st_size=0 even though boot_id contains a readable UUID.
  [ -r "$BOOT_ID" ] || fail 'Kernel boot identity is unavailable.'
  value=$(cat "$BOOT_ID") || fail 'Kernel boot identity could not be read.'
  [ "${#value}" -eq 36 ] || fail 'Kernel boot identity is malformed.'
  case "$value" in ????????-????-????-????-????????????) ;; *) fail 'Kernel boot identity is malformed.';; esac
  case "$value" in *[!0-9a-f-]*) fail 'Kernel boot identity is malformed.';; esac
  printf '%s\n' "$value"
}
stop_prudynt() {
  "$SERVICE" stop || true
  attempts=0
  while pidof prudynt >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    [ "$attempts" -lt 20 ] || return 1
    sleep 1
  done
  # SDK teardown can outlive the pidfile. Never start another instance immediately.
  sleep 2
}
unmount_known() {
  if mounted; then
    [ "$(hash "$TARGET")" = "$FIXED_SHA" ] || fail 'Mounted Prudynt is not this repair; refusing to unmount it.'
    umount "$TARGET" || fail 'Unable to unmount the verified repair.'
  fi
  [ "$(hash "$TARGET")" = "$STOCK_SHA" ] || fail 'Stock image was not restored.'
}
rollback_internal() {
  mode=$1
  owned || fail 'No owned clock-fix package to roll back.'
  check_target
  current=$(hash "$TARGET")
  [ "$current" = "$FIXED_SHA" ] || [ "$current" = "$STOCK_SHA" ] || fail 'Current Prudynt belongs to another installation.'
  # The guard disables its own hook before any restart, so a cold recovery boots stock.
  [ "$mode" = manual ] || disable_hook
  if ! stop_prudynt; then
    write_atomic "$PKG/rollback-failure" 'Prudynt did not exit; no unmount or duplicate warm start attempted.'
    if [ "$mode" != manual ]; then
      say 'First-boot rollback needs a cold reboot; owned hook is disabled.'
      sync
      reboot
      exit 1
    fi
    fail 'Prudynt did not exit. Patch left mounted; manual cold recovery is required.'
  fi
  unmount_known
  disable_hook
  # Only this owned package/runtime directory is removed. ROM and settings stay intact.
  rm -rf "$PKG"
  if [ -f "$RUN/owner" ] && [ "$(cat "$RUN/owner")" = "$OWNER" ]; then rm -rf "$RUN"; fi
  if [ "$mode" != manual ]; then
    say 'Verified stock image restored; rebooting cold with the owned hook disabled.'
    sync; reboot
    exit 0
  fi
  "$SERVICE" start
  say 'Stock binary restored and started; fresh video still needs verification. Use rollback --cold-reboot for a cold SDK recovery.' 
}
usage() {
  cat <<'USAGE'
Usage:
  install_prudynt_clockfix.sh dry-run --payload /tmp/prudynt-fixed.gz --sha256 HASH
  install_prudynt_clockfix.sh install --payload /tmp/prudynt-fixed.gz --sha256 HASH
  install_prudynt_clockfix.sh status
  /etc/opc-clockfix/control accept-boot
  /etc/opc-clockfix/control rollback [--cold-reboot]
Install only persists the verified package; it does not restart or remount a running streamer.
The first boot has a 180-second rollback guard. Accept only after independent RTP/recording checks.
USAGE
}
command=${1:-}
[ -n "$command" ] || { usage; exit 2; }
shift
case "$command" in
  dry-run|install)
    payload=
    declared_sha=
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --payload) [ "$#" -ge 2 ] || fail 'Missing payload path.'; payload=$2; shift 2;;
        --sha256) [ "$#" -ge 2 ] || fail 'Missing image hash.'; declared_sha=$2; shift 2;;
        *) fail "Unknown argument: $1";;
      esac
    done
    [ -n "$payload" ] && [ -f "$payload" ] || fail 'A gzip payload is required.'
    [ "$declared_sha" = "$FIXED_SHA" ] || fail 'Only the verified clock-fix image is accepted.'
    need_tools; check_target; lock
    command -v nohup >/dev/null 2>&1 || fail 'nohup is required for the first-boot guard.'
    command -v reboot >/dev/null 2>&1 || fail 'reboot is required for cold recovery.'
    STAGE=$(mktemp -d "$R/tmp/opc-clockfix-check.XXXXXX")
    expand_checked "$payload" "$STAGE/prudynt"
    if [ "$command" = install ] && { [ -e "$PKG" ] || [ -L "$PKG" ]; }; then
      check_package
      if ! check_hook; then
        [ ! -e "$HOOK" ] && [ ! -L "$HOOK" ] || fail 'An unrelated boot hook exists.'
        [ "$(free_kib "$R/etc")" -ge "$((RESERVE_KIB + 8))" ] || fail 'Insufficient reserve to restore the owned boot hook.'
      fi
      install_hook
      say 'Identical owned package already installed; running streamer unchanged.'
      exit 0
    fi
    payload_bytes=$(wc -c < "$payload" | tr -d ' ')
    incoming_gzip_sha=$(hash "$payload")
    control_bytes=$(wc -c < "$0" | tr -d ' ')
    required_kib=$(((payload_bytes + control_bytes + 1023) / 1024 + RESERVE_KIB + 32))
    available_kib=$(free_kib "$R/etc")
    [ "$available_kib" -ge "$required_kib" ] || fail "Overlay needs ${required_kib}KiB including reserve; only ${available_kib}KiB free."
    if [ "$command" = dry-run ]; then say "Preflight passed; requires ${required_kib}KiB including ${RESERVE_KIB}KiB reserve. No persistent files changed."; exit 0; fi
    if [ -e "$HOOK" ] || [ -L "$HOOK" ]; then fail 'An unrelated boot hook exists.'; fi
    rm -rf "$STAGE"
    STAGE=$PKG.stage.$$
    mkdir -m 700 "$STAGE"
    cp "$payload" "$STAGE/prudynt.gz"
    [ "$(hash "$STAGE/prudynt.gz")" = "$incoming_gzip_sha" ] || fail 'Payload changed while installing.'
    cp "$0" "$STAGE/control"
    chmod 600 "$STAGE/prudynt.gz"
    chmod 755 "$STAGE/control"
    printf '%s\n' "$OWNER" > "$STAGE/owner"
    printf '%s\n' "$FIXED_SHA" > "$STAGE/sha256"
    hash "$STAGE/prudynt.gz" > "$STAGE/gzip.sha256"
    hash "$STAGE/control" > "$STAGE/control.sha256"
    printf '%s\n' '#!/bin/sh' '# OpenPetCam owned clock-fix boot hook.' "case \"\${1:-}\" in" " start) $PKG/control boot || true;;" " stop) $PKG/control stop || true;;" 'esac' > "$STAGE/hook"
    chmod 755 "$STAGE/hook"
    hash "$STAGE/hook" > "$STAGE/hook.sha256"
    printf '%s\n' 'First reboot is not yet accepted.' > "$STAGE/boot.pending"
    mv "$STAGE" "$PKG"; STAGE=
    install_hook
    sync
    say 'Verified compressed package installed. Running streamer was not changed. Reboot starts the guarded trial.'
    ;;
  boot)
    need_tools; check_target; check_package; lock
    boot=$(boot_identity)
    if [ -f "$PKG/boot.pending" ]; then
      command -v nohup >/dev/null 2>&1 || fail 'nohup is required for the first-boot guard.'
      command -v reboot >/dev/null 2>&1 || fail 'reboot is required for cold recovery.'
    fi
    pidof prudynt >/dev/null 2>&1 && fail 'Streamer is already running; refusing a live remount.'
    current=$(hash "$TARGET")
    [ "$current" = "$STOCK_SHA" ] || [ "$current" = "$FIXED_SHA" ] || fail 'Current Prudynt identity differs.'
    [ "$(free_kib "$R/tmp")" -ge "$((FIXED_BYTES / 1024 + 1152))" ] || fail 'Insufficient RAM filesystem space.'
    if [ -e "$RUN" ]; then
      [ -f "$RUN/owner" ] && [ "$(cat "$RUN/owner")" = "$OWNER" ] || fail 'Unrelated runtime directory exists.'
    else mkdir -m 700 "$RUN"; printf '%s\n' "$OWNER" > "$RUN/owner"; fi
    IMAGE_TMP=$RUN/prudynt.new.$$
    expand_checked "$PKG/prudynt.gz" "$IMAGE_TMP"
    mv "$IMAGE_TMP" "$RUN/prudynt"
    IMAGE_TMP=
    if ! mounted; then
      mount -o bind "$RUN/prudynt" "$TARGET" || fail 'Bind mount failed; stock image remains.'
      BOOT_MOUNTED=1
    fi
    [ "$(hash "$TARGET")" = "$FIXED_SHA" ] || fail 'Bind-mounted image identity differs.'
    write_atomic "$RUN/boot" "$boot"
    if [ -f "$PKG/boot.pending" ]; then
      nohup "$PKG/control" guard "$boot" > "$RUN/guard.log" 2>&1 < /dev/null &
      NEW_GUARD_PID=$!
      printf '%s\n' "$NEW_GUARD_PID" > "$RUN/guard.pid"
    fi
    BOOT_SAFE=1
    say 'Verified RAM image mounted; stock S31prudynt can start it.'
    ;;
  guard)
    expected_boot=${1:-}
    [ -n "$expected_boot" ] || fail 'Missing guard boot identity.'
    sleep "$GUARD_SECONDS"
    [ -f "$PKG/boot.pending" ] || exit 0
    [ -f "$RUN/boot" ] && [ "$(cat "$RUN/boot")" = "$expected_boot" ] || exit 0
    [ "$(boot_identity)" = "$expected_boot" ] || exit 0
    waits=0
    until mkdir "$LOCK" 2>/dev/null; do
      [ -f "$PKG/boot.pending" ] || exit 0
      waits=$((waits + 1))
      if [ "$waits" -ge 30 ]; then
        disable_hook
        say 'First-boot guard could not acquire its operation lock; cold stock recovery required.' >&2
        sync; reboot
        exit 1
      fi
      sleep 1
    done
    LOCKED=1
    [ -f "$PKG/boot.pending" ] || exit 0
    say 'First boot was not accepted; restoring stock.'
    rollback_internal guard
    ;;
  accept-boot)
    need_tools; check_target; check_package; lock
    [ -f "$RUN/boot" ] && [ "$(cat "$RUN/boot")" = "$(boot_identity)" ] || fail 'The guarded boot has not run in this boot session.'
    mounted && [ "$(hash "$TARGET")" = "$FIXED_SHA" ] || fail 'The verified repair is not active.'
    pidof prudynt >/dev/null 2>&1 || fail 'Prudynt is not running.'
    write_atomic "$PKG/boot.accepted" "$(boot_identity) $FIXED_SHA"
    rm -f "$PKG/boot.pending"
    sync
    say 'This boot accepted. Future compatible boots start the verified repair automatically.'
    ;;
  rollback)
    mode=manual
    if [ "$#" -eq 1 ] && [ "$1" = --cold-reboot ]; then mode=cold; shift; fi
    [ "$#" -eq 0 ] || fail 'Unknown rollback option.'
    need_tools; lock; rollback_internal "$mode"
    ;;
  stop)
    need_tools; owned || exit 0; lock
    pidof prudynt >/dev/null 2>&1 && fail 'Streamer is still running; leaving bind mount alone.'
    unmount_known
    ;;
  status)
    [ "$#" -eq 0 ] || fail 'Unexpected status argument.'
    if owned; then
      say "Package installed; current image $(hash "$TARGET")."
      if [ -f "$PKG/boot.pending" ]; then say 'First boot awaits independent validation.'; else say 'Boot validation accepted.'; fi
    else say 'No owned package installed.'; fi
    ;;
  *) usage; exit 2;;
esac
