# Contributing to OpenPetCam

Thanks for helping. OpenPetCam is built by pet owners for their own pets, and every improvement reaches every
camera through the nightly updates. You don't need a camera to contribute: most of the code has simulations.

## Five minutes to a working checkout

```bash
git clone https://github.com/crimeacs/OpenPetCam.git && cd OpenPetCam
git config core.hooksPath .githooks          # the privacy guard runs on every commit
python3 -m pip install numpy pillow pyflakes
for t in tracker events presence stall disconnect pill search_here_first update pet_name theme; do
  PETCAM_SITE_ENV=config/site.env.example python3 tests/server/${t}_test.py
done
```

All of those run on a laptop with no camera or server: the tracker, presence and search tests drive the real state
machines through simulated rooms. CI runs the same list on every push and pull request.

## Where things are

| You want to change | Look in |
|---|---|
| What the phone app shows | `server/index.html` (one file: HTML, CSS and JS), themes in `server/themes.css` |
| How it finds and follows the pet | `server/tracker.py`, tested by `tests/server/tracker_test.py` |
| Detection, presence, sounds, clips | `server/watch.py` (`server/detector.py` runs the models) |
| The web server and API | `server/app.py` |
| Install and updates | `server/install.sh`, `server/update.py`, `deploy.sh`, `setup` |
| The self-improving agent | `tools/superagent.py`, `tools/report_agent.py`, `agent/` |

[docs/architecture.md](docs/architecture.md) explains how the pieces fit.

## Making a change

1. Write a test that fails without your change. Most bugs can be reproduced in a simulation
   (`tests/server/presence_test.py` is the pattern).
2. Keep the change small and say in the pull request what you measured before and after.
3. Run the tests above and `tools/check_public.sh --all`.
4. Open a pull request. The template asks for the few things a reviewer needs.

Changes that move the camera or make sounds near a pet need a note about noise and frequency: pets get startled,
and the defaults stay conservative on purpose.

## Privacy rules (the guard enforces most of them)

- Never commit frames, clips, reports, labels, IP addresses, host names, e-mail addresses, people's names or your
  pet's name. Those live in `config/` and `eval/`, which are git-ignored.
- `tools/check_public.sh` refuses commits containing your server's address, your camera's address, secret-shaped
  strings or your pet's name (taken from your `config/site.env`). Don't bypass it.
- In issues, describe what happened in words; don't paste report contents or screenshots of your home unless you
  mean to share them.

## Pull requests opened by agents

If you run the optional agent (`tools/install_autopilot.sh`), fixes it proves on your own install are offered here
as pull requests by `tools/contribute.py`, with a public summary and the checks they passed. They are reviewed like
any other pull request. If you'd rather keep your fixes local, set `CONTRIBUTE=off` in `config/site.env`.

## Releases

Maintainers add a `## [X.Y.Z]` section to `CHANGELOG.md` and push a `vX.Y.Z` tag. The release workflow runs the
checks and publishes the archive that servers install overnight.
