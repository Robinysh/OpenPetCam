"""Execute the production privacy gate with isolated filesystem and service doubles.

The reviewed fixture hashes are substituted in a temporary copy. No real camera,
service, mount, PID, network, reboot or power-control command is used.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/firmware/install_privacy_guard.sh'
STOCK = b'fixture stock executable'
FIXED = b'fixture fixed executable'
SDK = b'fixture SDK'
SERVICE = b'''#!/bin/sh
set -eu
r=$OPC_PRIVACY_TEST_ROOT
printf 'stock noise: %s\\n' "$1"
echo "$1" >> "$r/tmp/actions"
case "$1" in
 start)
  if [ -f "$r/tmp/spawn-fail" ]; then touch "$r/tmp/running"; exit 1; fi
  [ ! -f "$r/tmp/fail-start" ] || exit 1
  touch "$r/tmp/running";;
 stop)
  if [ -e "$r/etc/opc-privacy/paused" ]; then echo 'marker-before-stop' >> "$r/tmp/actions"; fi
  [ -f "$r/tmp/hang" ] || rm -f "$r/tmp/running"
  [ ! -f "$r/tmp/stop-error" ] || exit 1;;
 *) exit 2;;
esac
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PrivacyCameraTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opc-privacy-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ['etc/init.d', 'rom/etc/init.d', 'rom/usr/bin', 'usr/bin', 'usr/lib', 'tmp', 'bin']:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / '.opc-privacy-test-root').touch()
        for name, content in [('rom/usr/bin/prudynt', STOCK), ('usr/bin/prudynt', FIXED),
                              ('usr/lib/libimp.so', SDK), ('rom/etc/init.d/S31prudynt', SERVICE),
                              ('etc/init.d/S31prudynt', SERVICE)]:
            (self.root / name).write_bytes(content)
        (self.root / 'etc/init.d/S31prudynt').chmod(0o755)
        self.clockfix = self.root / 'etc/init.d/S30opc-clockfix'
        self.clockfix.write_bytes(b'existing clockfix sentinel')
        script = SCRIPT.read_text()
        for old, new in [('5ec4c4315f02b68a3a0ee01b539fb3fc1af54d7dabd875a58709da4e06e53907', digest(SERVICE)),
                         ('3f23b94eab5463db62c069e83dd48cc535ce2d91b6990f216cb0efb587f9d25d', digest(STOCK)),
                         ('45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b', digest(FIXED)),
                         ('bd904ec8f4123965ded8c0c57c25ad5c3ae0fdc17aa4a6374278a7c822cb31d9', digest(SDK))]:
            self.assertIn(old, script)
            script = script.replace(old, new)
        self.installer = self.root / 'tmp/installer.sh'
        self.installer.write_text(script)
        self.installer.chmod(0o755)
        self.env = dict(os.environ, OPC_PRIVACY_TEST_ROOT=str(self.root),
                        PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'])
        self.wrapper('df', '''echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'
free=392; [ ! -f "$OPC_PRIVACY_TEST_ROOT/tmp/low-space" ] || free=1
echo "mock 960 568 $free 59% /"''')
        self.wrapper('pidof', '''[ -f "$OPC_PRIVACY_TEST_ROOT/tmp/running" ] || exit 1
echo 99999''')
        self.wrapper('sleep', 'echo "sleep $*" >> "$OPC_PRIVACY_TEST_ROOT/tmp/actions"')
        self.wrapper('sync', ':')
        self.wrapper('rmdir', 'exit 127')
        (self.root / 'tmp/running').touch()

    def wrapper(self, name, body):
        p = self.root / 'bin' / name
        p.write_text('#!/bin/sh\nset -eu\n' + body + '\n')
        p.chmod(0o755)

    @property
    def pkg(self):
        return self.root / 'etc/opc-privacy'

    @property
    def marker(self):
        return self.pkg / 'paused'

    @property
    def service(self):
        return self.root / 'etc/init.d/S31prudynt'

    def run_script(self, *args, control=False, service=False, ok=True):
        script = self.service if service else self.pkg / 'control' if control else self.installer
        r = subprocess.run([str(script), *args], env=self.env, capture_output=True, text=True, timeout=10)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def install(self):
        return self.run_script('install')

    def status(self):
        return json.loads(self.run_script('status', control=True).stdout)

    def actions(self):
        p = self.root / 'tmp/actions'
        return p.read_text().splitlines() if p.exists() else []

    def test_explicit_command_required_and_test_root_must_be_marked(self):
        self.assertNotEqual(self.run_script(ok=False).returncode, 0)
        (self.root / '.opc-privacy-test-root').unlink()
        self.assertNotEqual(self.run_script('install', ok=False).returncode, 0)
        self.assertFalse(self.pkg.exists())
        self.assertEqual(self.service.read_bytes(), SERVICE)

    def test_preflight_does_not_stop_capture_or_persist_any_files(self):
        self.run_script('dry-run')
        self.assertEqual(self.service.read_bytes(), SERVICE)
        self.assertFalse(self.pkg.exists())
        self.assertTrue((self.root / 'tmp/running').exists())
        self.assertEqual(self.actions(), [])
        self.assertFalse((self.root / 'tmp/opc-privacy.lock').exists())

    def test_install_is_idempotent_and_does_not_touch_capture_or_clockfix(self):
        self.install(); self.install()
        self.assertEqual((self.pkg / 'service.original').read_bytes(), SERVICE)
        self.assertNotEqual(self.service.read_bytes(), SERVICE)
        self.assertTrue((self.root / 'tmp/running').exists())
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.actions(), [])
        self.assertEqual(self.clockfix.read_bytes(), b'existing clockfix sentinel')
        self.assertEqual(self.status(), {'mode': 'capture_running', 'privacy': False, 'paused': False,
                                         'running': True, 'verified': True, 'boot_persistent': True})

    def test_foreign_service_package_or_sdk_is_not_overwritten(self):
        for path in [self.service, self.root / 'usr/lib/libimp.so', self.root / 'rom/etc/init.d/S31prudynt']:
            original = path.read_bytes(); path.write_bytes(b'foreign')
            self.assertNotEqual(self.run_script('install', ok=False).returncode, 0)
            self.assertFalse(self.pkg.exists())
            self.assertEqual(path.read_bytes(), b'foreign')
            path.write_bytes(original)
        self.pkg.mkdir(); (self.pkg / 'foreign').write_text('data')
        self.assertNotEqual(self.run_script('install', ok=False).returncode, 0)
        self.assertEqual((self.pkg / 'foreign').read_text(), 'data')

    def test_low_overlay_space_fails_before_service_replacement(self):
        (self.root / 'tmp/low-space').touch()
        self.assertNotEqual(self.run_script('install', ok=False).returncode, 0)
        self.assertEqual(self.service.read_bytes(), SERVICE)
        self.assertFalse(self.pkg.exists())
        self.assertEqual(self.actions(), [])

    def test_pause_persists_marker_before_stop_and_returns_only_verified_json(self):
        self.install()
        result = json.loads(self.run_script('pause', control=True).stdout)
        self.assertEqual(result, {'mode': 'capture_stopped', 'privacy': True, 'paused': True,
                                  'running': False, 'verified': True, 'boot_persistent': True})
        self.assertTrue(self.marker.exists())
        self.assertFalse((self.root / 'tmp/running').exists())
        self.assertEqual(self.actions(), ['stop', 'marker-before-stop', 'sleep 2'])
        self.assertEqual(self.clockfix.read_bytes(), b'existing clockfix sentinel')

    def test_paused_start_and_restart_cannot_reopen_capture(self):
        self.install(); self.run_script('pause', control=True)
        self.run_script('start', service=True)
        self.run_script('restart', service=True)
        self.assertFalse((self.root / 'tmp/running').exists())
        self.assertNotIn('start', self.actions())
        self.assertTrue(self.marker.exists())

    def test_paused_state_survives_simulated_boot_and_stops_unexpected_capture(self):
        self.install(); self.run_script('pause', control=True)
        # Only tmpfs/runtime state is lost on boot; durable /etc flag remains.
        import shutil
        shutil.rmtree(self.root / 'tmp/opc-privacy')
        self.run_script('start', service=True)
        self.assertFalse((self.root / 'tmp/running').exists())
        (self.root / 'tmp/running').touch()
        self.run_script('start', service=True)
        self.assertFalse((self.root / 'tmp/running').exists())
        self.assertNotIn('start', self.actions())

    def test_resume_requires_explicit_request_and_does_not_duplicate_running_process(self):
        self.install(); self.run_script('pause', control=True)
        resumed = json.loads(self.run_script('resume', control=True).stdout)
        self.assertEqual(resumed['mode'], 'capture_running')
        self.assertFalse(resumed['privacy'])
        self.assertTrue(resumed['verified'])
        self.assertFalse(self.marker.exists())
        self.run_script('resume', control=True)
        self.run_script('start', service=True)
        self.assertEqual(self.actions().count('start'), 1)

    def test_start_failure_restores_persistent_pause_and_cleans_partial_process(self):
        self.install(); self.run_script('pause', control=True)
        for flag in ['fail-start', 'spawn-fail']:
            f = self.root / 'tmp' / flag; f.touch()
            self.assertNotEqual(self.run_script('resume', control=True, ok=False).returncode, 0)
            self.assertTrue(self.marker.exists())
            self.assertFalse((self.root / 'tmp/running').exists())
            self.assertEqual(self.status()['mode'], 'capture_stopped')
            f.unlink()

    def test_resume_precondition_failure_does_not_clear_persistent_pause(self):
        self.install(); self.run_script('pause', control=True)
        sdk = self.root / 'usr/lib/libimp.so'; original = sdk.read_bytes()
        sdk.write_bytes(b'unreviewed SDK')
        self.assertNotEqual(self.run_script('resume', control=True, ok=False).returncode, 0)
        self.assertTrue(self.marker.exists())
        self.assertFalse((self.root / 'tmp/running').exists())
        sdk.write_bytes(original)
        import shutil
        runtime = self.root / 'tmp/opc-privacy'; shutil.rmtree(runtime); runtime.mkdir()
        self.assertNotEqual(self.run_script('resume', control=True, ok=False).returncode, 0)
        self.assertTrue(self.marker.exists())
        self.assertFalse((self.root / 'tmp/running').exists())

    def test_hung_stop_keeps_durable_pause_and_never_reports_verified(self):
        self.install(); (self.root / 'tmp/hang').touch()
        self.assertNotEqual(self.run_script('pause', control=True, ok=False).returncode, 0)
        self.assertTrue(self.marker.exists())
        self.assertTrue((self.root / 'tmp/running').exists())
        result = self.status()
        self.assertTrue(result['privacy'])
        self.assertFalse(result['verified'])
        self.assertEqual(result['mode'], 'capture_unavailable')
        self.assertNotIn('start', self.actions())

    def test_stop_return_error_is_not_mistaken_for_capture_still_running(self):
        self.install(); (self.root / 'tmp/stop-error').touch()
        result = json.loads(self.run_script('pause', control=True).stdout)
        self.assertTrue(result['verified'])
        self.assertFalse(result['running'])

    def test_gate_or_control_corruption_fails_closed_without_replacing_foreign_files(self):
        self.install(); gate = self.service.read_bytes()
        self.service.write_text('foreign gate')
        self.assertNotEqual(self.run_script('pause', control=True, ok=False).returncode, 0)
        self.assertEqual(self.service.read_text(), 'foreign gate')
        self.assertFalse(self.marker.exists())
        self.service.write_bytes(gate)
        with (self.pkg / 'control').open('a') as out:
            out.write('\n# corrupted identity\n')
        self.assertNotEqual(self.run_script('start', service=True, ok=False).returncode, 0)
        self.assertEqual(self.actions(), [])

    def test_uninstall_refuses_while_paused_and_preserves_unrelated_clockfix(self):
        self.install(); self.run_script('pause', control=True)
        self.assertNotEqual(self.run_script('uninstall', control=True, ok=False).returncode, 0)
        self.assertTrue(self.marker.exists())
        self.assertNotEqual(self.service.read_bytes(), SERVICE)
        self.run_script('resume', control=True)
        before = self.actions()
        self.run_script('uninstall', control=True)
        self.assertEqual(self.service.read_bytes(), SERVICE)
        self.assertFalse(self.pkg.exists())
        self.assertTrue((self.root / 'tmp/running').exists())
        self.assertEqual(self.actions(), before)
        self.assertEqual(self.clockfix.read_bytes(), b'existing clockfix sentinel')

    def test_foreign_lock_is_never_removed(self):
        self.install(); lock = self.root / 'tmp/opc-privacy.lock'; lock.mkdir()
        (lock / 'other').write_text('held')
        self.assertNotEqual(self.run_script('pause', control=True, ok=False).returncode, 0)
        self.assertEqual((lock / 'other').read_text(), 'held')
        self.assertFalse(self.marker.exists())

    def test_corrupt_marker_contents_still_block_boot(self):
        self.install(); self.run_script('pause', control=True)
        self.marker.write_text('0\n')
        self.run_script('start', service=True)
        self.assertTrue(self.status()['privacy'])
        self.assertNotIn('start', self.actions())

    def test_install_refuses_implicit_version_upgrade(self):
        self.install()
        with self.installer.open('a') as out:
            out.write('\n# another incoming version\n')
        self.assertNotEqual(self.run_script('install', ok=False).returncode, 0)
        self.assertTrue((self.root / 'tmp/running').exists())


if __name__ == '__main__':
    unittest.main()
