# Secrets and privacy

| What | Where | In git? |
|---|---|---|
| Pet name, server address, time zone, SSH settings | `config/site.env` on your computer; non-SSH parts copied to `/etc/petcam/site.env` | no (git-ignored) |
| Web app password (PBKDF2 hash only), session key, camera web login | `/etc/petcam/app.env` on the server, mode 640, set by `set-secrets.sh` | no, never leaves the server |
| Camera RTSP login | `/etc/petcam/camera.env` | no |
| Web push keys, WireGuard private key | generated on the server by `install.sh` | no |
| Frames, crops, labels and reports from your home | `eval/` on your computer | no (git-ignored) |

Enable the guard once per clone: `git config core.hooksPath .githooks`. The pre-commit hook
(`tools/check_public.sh`) refuses any commit that contains your server's IP or host name, your camera's address,
a private key or other secret-shaped strings. Run `tools/check_public.sh --all` before publishing a fork.

Phones stay signed in with a device cookie; the server stores only its SHA-256 hash, and a device can be revoked
from Settings. Login attempts are rate-limited.

Privacy pauses new capture. On the verified Thingino T23 installation, an owned
camera-side guard stops Prudynt, the video/audio capture and streaming process.
Its pause marker survives reboot and blocks service start/restart attempts until
you explicitly resume. The server also stops its listeners, detection and new
observations. Existing recordings remain available.

The app clears the live picture and closes its live connection immediately. It
shows a confirmed pause only after the camera reports the stopped process and
persistent gate. If confirmation fails, live access remains paused and the app
shows that privacy needs attention. Resume requires the camera to confirm its
picture and microphone settings before acquisition resumes. A paused camera does
not trigger the watcher's automatic restart/reboot recovery or ordinary missing
recording alerts.

On cameras without this guard, the legacy mode blacks out the picture and
switches off the microphone; the app identifies that weaker mode explicitly.
Once the guard has been seen, a missing or broken guard is an error, never a
reason to silently fall back. The image-specific installer is
`tools/firmware/install_privacy_guard.sh`; it preserves the reviewed original
service and does not change the running state during installation.

The camera stays powered and reachable so it can resume remotely. This is
software privacy, not electrical isolation of the image sensor or microphone.
It cannot protect against compromised camera firmware. A physical disconnect
requires unplugging the camera or adding a separately controlled power switch.
