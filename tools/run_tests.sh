#!/bin/bash
# The one offline test gate, used by CI (.github/workflows/ci.yml) and by the release workflow, so a release can
# never pass a smaller gate than main. Needs no camera, no server and no private config.
#   tools/run_tests.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PETCAM_SITE_ENV=config/site.env.example       # the server code's settings
export PETCAM_CONFIG=config/site.env.example         # tools/siteconf.py (the simulator's camera tools import it)
# the same lint CI runs first (undefined names, unused imports)
if python3 -c "import pyflakes" 2>/dev/null; then echo "== pyflakes"; python3 -m pyflakes server tools tests sim; fi
for t in tracker events presence stall disconnect pill search_here_first search_roommap install_files no_rotation room roomfollow update pet_name theme search_evidence \
         follow search_cancel spot_highlight talk_button policy fast_mode search_glimpse install_retire \
         clip_player coverage zones search_overshoot js_syntax flap agent_loop fasttrack chase_wall remount; do
  echo "== $t"
  timeout 300 python3 "tests/server/${t}_test.py"
done
echo "== sim"
timeout 600 python3 tests/sim/sim_test.py
timeout 300 python3 tests/sim/sysid_recovery_test.py
timeout 300 python3 tests/sim/ab_report_test.py
timeout 120 python3 tests/sim/world_test.py
timeout 120 python3 tests/sim/acoustic_geometry_test.py
timeout 120 python3 tests/sim/acoustic_waveforms_test.py
timeout 120 python3 tests/sim/acoustic_fit_test.py
echo "all offline tests passed"
