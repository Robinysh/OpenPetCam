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
