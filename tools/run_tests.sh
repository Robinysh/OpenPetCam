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
         agent_backlog agent_validation hard_examples hard_examples_ui train_detector goal_monitor goal_monitor_api goal_monitor_ui startup_metrics startup_goal_projection startup_goal_ui monitoring_navigation viewstats view_insights_api view_insights_ui activity_history follow_metrics report_api report_evidence requestboard requests_history_frontend request_actions_ui live_ui_regression build_reload_ui \
         clip_media clip_acquisition clip_player_ui clip_warning_ui video_clock prudynt_timestamp privacy privacy_camera privacy_watch privacy_alert privacy_ui soundfind peek_step report_frame search_phase search_after_find look_after_privacy still_hold still_hold_bounds search_snap check_public follow_delay native_recheck rpi_camera pet_kind; do
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
timeout 120 python3 tests/sim/camera_calibration_test.py
timeout 120 python3 tests/sim/calibration_validation_test.py
timeout 120 python3 tests/sim/nir_reflectance_test.py
timeout 120 python3 tests/sim/nir_capture_test.py
timeout 120 python3 tests/sim/nir_fusion_test.py
timeout 120 python3 tests/sim/nir_sensor_test.py
timeout 120 python3 tests/sim/nir_ui_test.py
timeout 120 python3 tests/sim/search_routes_test.py
timeout 120 python3 tests/sim/search_prior_test.py
timeout 120 python3 tests/sim/search_scene_test.py
timeout 120 python3 tests/sim/search_bench_test.py
timeout 120 python3 tests/sim/search_report_test.py
timeout 120 python3 tests/sim/view_obstructions_test.py
timeout 120 python3 tests/sim/obstruction_scene_test.py
timeout 120 python3 tests/sim/obstruction_model_test.py
timeout 120 python3 tests/sim/obstruction_forest_test.py
timeout 120 python3 tests/sim/obstruction_reference_test.py
timeout 120 python3 tests/sim/obstruction_bench_test.py
timeout 120 python3 tests/sim/obstruction_report_test.py
timeout 120 python3 tests/sim/acoustic_localization_model_test.py
timeout 120 python3 tests/sim/acoustic_localization_test.py
python3 -c 'import shutil; assert shutil.which("clang++") or shutil.which("c++"), "3D acoustic physics checks require a C++ compiler"'
timeout 120 python3 tests/sim/acoustic_wave3d_test.py
timeout 120 python3 tests/sim/acoustic_phase_localization_test.py
timeout 120 python3 tests/sim/acoustic_wave_room_test.py
timeout 120 python3 tests/sim/acoustic_wave_experiment_test.py
timeout 120 python3 tests/sim/acoustic_wave_report_test.py
timeout 120 python3 tests/sim/acoustic_geometry_test.py
timeout 120 python3 tests/sim/acoustic_waveforms_test.py
timeout 120 python3 tests/sim/acoustic_fit_test.py
timeout 120 python3 tests/sim/acoustic_probes_test.py
timeout 120 python3 tests/sim/acoustic_native_rate_test.py
timeout 120 python3 tests/sim/acoustic_acquisition_rate_policy_test.py
timeout 120 python3 tests/sim/acoustic_shot_qc_test.py
timeout 120 python3 tests/sim/acoustic_survey_test.py
timeout 120 python3 tests/sim/acoustic_capture_test.py
timeout 120 python3 tests/sim/acoustic_adapter_test.py
timeout 120 python3 tests/sim/acoustic_gathers_test.py
timeout 120 python3 tests/sim/acoustic_volume_test.py
timeout 120 python3 tests/sim/acoustic_device_native_test.py
timeout 120 python3 tests/sim/acoustic_device_io_test.py
timeout 120 python3 tests/sim/acoustic_device_decode_test.py
timeout 120 python3 tests/sim/acoustic_localization_adaptive_test.py
timeout 120 python3 tests/sim/acoustic_localization_bearing_test.py
timeout 120 python3 tests/sim/acoustic_localization_data_test.py
timeout 120 python3 tests/sim/acoustic_localization_development_test.py
timeout 120 python3 tests/sim/acoustic_localization_exact_oracle_test.py
timeout 120 python3 tests/sim/acoustic_localization_experiment_test.py
timeout 120 python3 tests/sim/acoustic_localization_fresh_stress_test.py
timeout 120 python3 tests/sim/acoustic_localization_harmonic_test.py
timeout 120 python3 tests/sim/acoustic_localization_identifiability_test.py
timeout 120 python3 tests/sim/acoustic_localization_pipeline_test.py
timeout 120 python3 tests/sim/acoustic_localization_real_test.py
timeout 120 python3 tests/sim/acoustic_localization_resolution_test.py
timeout 120 python3 tests/sim/acoustic_localization_sensitivity_test.py
timeout 120 python3 tests/sim/world_acoustic_room_test.py
timeout 120 python3 tests/sim/world_mount_geometry_test.py
echo "all offline tests passed"
