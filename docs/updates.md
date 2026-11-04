# Updates

Once a day (around 04:20 server time) the server checks this repository's
[releases](https://github.com/crimeacs/OpenPetCam/releases). When there is a newer one it:
1. downloads it and checks its SHA-256;
2. keeps a copy of what is running;
3. installs it with the same `install.sh` that `./deploy.sh` uses;
4. checks that the app answers and the watcher sees pictures again within 3 minutes;
5. puts the previous version back if not, and never retries that release.

Settings in the app shows the version and the last check. Your settings, passwords, keys, trained pet model,
recordings and history are not part of a release, so an update never touches them.

- **Turn it off:** `UPDATE_CHANNEL=off` in `config/site.env`, then `./deploy.sh`. The app still tells you when a
  release is out.
- **Your own fork:** `UPDATE_REPO=you/your-fork`.
- **Running your own changes:** a server deployed from a checkout that is not exactly a release tag (for example
  `v0.1.0-3-gabc1234`, or with uncommitted changes) is never overwritten automatically. `./deploy.sh` updates it.
- **Publishing a release** (maintainers): add a `## [X.Y.Z]` section to `CHANGELOG.md`, then
  `git tag vX.Y.Z && git push --tags`. The release workflow runs the checks and attaches the archive.
