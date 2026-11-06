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
         clip_player coverage zones search_overshoot js_syntax flap agent_loop fasttrack chase_wall remount moving scheduled_sweep fast_lost search_memory watch_evidence framing_sound pose_observation \
         agent_backlog agent_validation viewstats view_insights_api view_insights_ui activity_history follow_metrics report_api requestboard requests_history_frontend live_ui_regression build_reload_ui \
         clip_media clip_acquisition clip_player_ui clip_warning_ui video_clock prudynt_timestamp; do
  echo "== $t"
  timeout 300 python3 "tests/server/${t}_test.py"
done
echo "== prudynt installer"
timeout 120 python3 tools/firmware/prudynt-timestamps/install_helper_test.py
echo "== sim"
timeout 600 python3 tests/sim/sim_test.py
timeout 300 python3 tests/sim/sysid_recovery_test.py
timeout 300 python3 tests/sim/ab_report_test.py
timeout 120 python3 tests/sim/world_test.py
echo "all offline tests passed"
