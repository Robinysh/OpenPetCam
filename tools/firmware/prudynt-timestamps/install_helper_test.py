#!/usr/bin/env python3
"""Exercise the BusyBox-compatible installer against an isolated fake root.

Only immutable fixture hashes/sizes differ in the tested script copy. No real
mount, service, PID, reboot or camera command is allowed by these command doubles.
"""
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import threading
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'install_prudynt_clockfix.sh'
STOCK = b'fixture stock prudynt'
FIXED = b'fixture fixed prudynt binary'
SDK = b'fixture SDK libimp'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opc-install-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for path in ['etc/init.d', 'rom/usr/bin', 'usr/bin', 'usr/lib', 'tmp', 'proc/sys/kernel/random', 'bin']:
            (self.root / path).mkdir(parents=True, exist_ok=True)
        (self.root / '.opc-clockfix-test-root').touch()
        (self.root / 'rom/usr/bin/prudynt').write_bytes(STOCK)
        (self.root / 'usr/bin/prudynt').write_bytes(STOCK)
        (self.root / 'usr/lib/libimp.so').write_bytes(SDK)
        (self.root / 'proc/mounts').write_text('')
        (self.root / 'proc/sys/kernel/random/boot_id').write_text('00000000-0000-4000-8000-000000000001\n')
        self.payload = self.root / 'tmp/fixed.gz'
        self.payload.write_bytes(gzip.compress(FIXED))
        content = SCRIPT.read_text().replace('3f23b94eab5463db62c069e83dd48cc535ce2d91b6990f216cb0efb587f9d25d', digest(STOCK)).replace('bd904ec8f4123965ded8c0c57c25ad5c3ae0fdc17aa4a6374278a7c822cb31d9', digest(SDK)).replace('45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b', digest(FIXED)).replace('FIXED_BYTES=967000', f'FIXED_BYTES={len(FIXED)}')
        self.script = self.root / 'tmp/installer.sh'
        self.script.write_text(content)
        self.script.chmod(0o755)
        self.env = dict(os.environ, OPC_CLOCKFIX_TEST_ROOT=str(self.root), PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'])
        self.wrapper('df', '''echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'
free=700; [ ! -f "$OPC_CLOCKFIX_TEST_ROOT/tmp/low-space" ] || free=1
case "$*" in *"/tmp"*) free=17000;; esac
echo "mockfs 20000 1000 $free 5% /"''')
        self.wrapper('mount', '''cp "$3" "$4"
printf 'mock %s none rw,bind 0 0\\n' "$4" > "$OPC_CLOCKFIX_TEST_ROOT/proc/mounts"
echo mount >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"''')
        self.wrapper('umount', '''cp "$OPC_CLOCKFIX_TEST_ROOT/rom/usr/bin/prudynt" "$1"
: > "$OPC_CLOCKFIX_TEST_ROOT/proc/mounts"
echo umount >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"''')
        self.wrapper('pidof', '''if [ -f "$OPC_CLOCKFIX_TEST_ROOT/tmp/running" ]; then echo 99999; exit 0; fi
exit 1''')
        self.wrapper('sleep', 'echo "sleep $*" >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"')
        self.wrapper('sync', ':')
        self.wrapper('reboot', 'echo reboot >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"')
        self.wrapper('nohup', 'echo "guard $*" >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"')
        self.service = self.root / 'etc/init.d/S31prudynt'
        self.service.write_text('''#!/bin/sh
echo "service $1" >> "$OPC_CLOCKFIX_TEST_ROOT/tmp/actions"
case "$1" in
start) touch "$OPC_CLOCKFIX_TEST_ROOT/tmp/running";;
stop) [ -f "$OPC_CLOCKFIX_TEST_ROOT/tmp/hang" ] || rm -f "$OPC_CLOCKFIX_TEST_ROOT/tmp/running";;
esac
''')
        self.service.chmod(0o755)

    def wrapper(self, name, body):
        p = self.root / 'bin' / name
        p.write_text('#!/bin/sh\nset -eu\n' + body + '\n')
        p.chmod(0o755)

    @property
    def package(self):
        return self.root / 'etc/opc-clockfix'

    def run_script(self, command, check=True, control=False):
        path = self.package / 'control' if control else self.script
        args = [str(path), *command]
        result = subprocess.run(args, env=self.env, capture_output=True, text=True, timeout=10)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def install(self):
        return self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)])

    def actions(self):
        p = self.root / 'tmp/actions'
        return p.read_text() if p.exists() else ''

    def boot(self):
        self.run_script(['boot'], control=True)
        (self.root / 'tmp/running').touch()
        for _ in range(100):
            if 'guard' in self.actions():
                break
            time.sleep(.01)

    def test_missing_command_cannot_install_or_restart(self):
        self.assertNotEqual(self.run_script([], check=False).returncode, 0)
        self.assertFalse(self.package.exists())
        self.assertEqual(self.actions(), '')

    def test_dry_run_leaves_persistent_tree_and_active_streamer_unchanged(self):
        (self.root / 'tmp/running').touch()
        self.run_script(['dry-run', '--payload', str(self.payload), '--sha256', digest(FIXED)])
        self.assertFalse(self.package.exists())
        self.assertEqual(self.actions(), '')
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)

    def test_install_only_persists_owned_package_and_is_idempotent(self):
        (self.root / 'tmp/running').touch()
        self.install()
        self.assertTrue((self.package / 'boot.pending').exists())
        self.assertTrue((self.root / 'etc/init.d/S30opc-clockfix').exists())
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)
        self.assertEqual(self.actions(), '')
        (self.root / 'tmp/low-space').touch()
        self.install()
        self.assertEqual(self.actions(), '')

    def test_lock_cleanup_does_not_depend_on_rmdir(self):
        self.wrapper('rmdir', 'exit 127')
        self.run_script(['dry-run', '--payload', str(self.payload), '--sha256', digest(FIXED)])
        self.assertFalse((self.root / 'tmp/opc-clockfix.lock').exists())
        self.install()
        self.assertFalse((self.root / 'tmp/opc-clockfix.lock').exists())

    def test_wrong_rom_or_sdk_prevents_all_persistent_changes(self):
        for path in ['rom/usr/bin/prudynt', 'usr/lib/libimp.so']:
            p = self.root / path
            original = p.read_bytes(); p.write_bytes(b'wrong')
            result = self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)], check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.package.exists())
            p.write_bytes(original)

    def test_wrong_payload_and_insufficient_space_fail_before_hook(self):
        self.payload.write_bytes(gzip.compress(b'not the verified image'))
        result = self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)], check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.package.exists())
        self.payload.write_bytes(gzip.compress(FIXED)); (self.root / 'tmp/low-space').touch()
        self.assertNotEqual(self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)], check=False).returncode, 0)
        self.assertFalse(self.package.exists())
        self.assertFalse((self.root / 'etc/init.d/S30opc-clockfix').exists())

    def test_foreign_hook_or_package_is_not_replaced(self):
        hook = self.root / 'etc/init.d/S30opc-clockfix'; hook.write_text('foreign')
        self.assertNotEqual(self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)], check=False).returncode, 0)
        self.assertEqual(hook.read_text(), 'foreign')
        hook.unlink(); self.package.mkdir(); (self.package / 'unrelated').write_text('owner data')
        self.assertNotEqual(self.run_script(['install', '--payload', str(self.payload), '--sha256', digest(FIXED)], check=False).returncode, 0)
        self.assertEqual((self.package / 'unrelated').read_text(), 'owner data')

    def test_first_boot_binds_verified_ram_image_and_arms_guard(self):
        self.install(); self.boot()
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), FIXED)
        self.assertIn('mount', self.actions())
        self.assertIn('guard', self.actions())
        self.assertTrue((self.package / 'boot.pending').exists())

    def test_accept_requires_current_guarded_boot_and_active_fixed_image(self):
        self.install()
        self.assertNotEqual(self.run_script(['accept-boot'], control=True, check=False).returncode, 0)
        self.boot(); self.run_script(['accept-boot'], control=True)
        self.assertFalse((self.package / 'boot.pending').exists())
        before = self.actions().count('guard')
        (self.root / 'tmp/running').unlink()
        self.run_script(['boot'], control=True)
        self.assertEqual(self.actions().count('guard'), before)

    def test_expired_first_boot_restores_stock_without_duplicate_warm_start(self):
        self.install(); self.boot()
        self.run_script(['guard', '00000000-0000-4000-8000-000000000001'], control=True)
        self.assertFalse(self.package.exists())
        self.assertFalse((self.root / 'etc/init.d/S30opc-clockfix').exists())
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)
        actions = self.actions()
        self.assertLess(actions.index('service stop'), actions.index('sleep 2'))
        self.assertLess(actions.index('sleep 2'), actions.index('umount'))
        self.assertLess(actions.index('umount'), actions.index('reboot'))
        self.assertNotIn('service start', actions)

    def test_guard_only_reboots_after_disabling_hook_if_sdk_process_will_not_exit(self):
        self.install(); self.boot(); (self.root / 'tmp/hang').touch()
        self.assertNotEqual(self.run_script(['guard', '00000000-0000-4000-8000-000000000001'], control=True, check=False).returncode, 0)
        self.assertIn('reboot', self.actions())
        self.assertNotIn('umount', self.actions())
        self.assertNotIn('service start', self.actions())
        self.assertFalse((self.root / 'etc/init.d/S30opc-clockfix').exists())
        self.assertTrue((self.package / 'rollback-failure').exists())

    def test_manual_stop_timeout_never_reboots_or_unmounts(self):
        self.install(); self.boot(); (self.root / 'tmp/hang').touch()
        self.assertNotEqual(self.run_script(['rollback'], control=True, check=False).returncode, 0)
        self.assertNotIn('reboot', self.actions())
        self.assertNotIn('umount', self.actions())
        self.assertTrue((self.root / 'etc/init.d/S30opc-clockfix').exists())

    def test_boot_identity_failure_cannot_leave_a_bind_mount(self):
        self.install()
        (self.root / 'proc/sys/kernel/random/boot_id').unlink()
        self.assertNotEqual(self.run_script(['boot'], control=True, check=False).returncode, 0)
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)
        self.assertNotIn('mount', self.actions())

    def test_readable_zero_stat_size_boot_identity_is_accepted(self):
        self.install()
        identity = self.root / 'proc/sys/kernel/random/boot_id'
        identity.unlink(); os.mkfifo(identity)
        self.assertEqual(identity.stat().st_size, 0)
        def publish():
            with identity.open('w') as stream:
                stream.write('00000000-0000-4000-8000-000000000001\n')
        writer = threading.Thread(target=publish, daemon=True)
        writer.start()
        self.run_script(['boot'], control=True)
        writer.join(2)
        self.assertFalse(writer.is_alive())
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), FIXED)

    def test_post_mount_guard_state_failure_unmounts_before_init_can_start(self):
        self.install()
        runtime = self.root / 'tmp/opc-clockfix'
        runtime.mkdir()
        (runtime / 'owner').write_text('openpetcam-prudynt-clockfix-v1\n')
        (runtime / 'guard.pid').mkdir()
        self.assertNotEqual(self.run_script(['boot'], control=True, check=False).returncode, 0)
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)
        self.assertIn('umount', self.actions())

    def test_explicit_manual_cold_rollback_disables_hook_and_reboots_stock(self):
        self.install(); self.boot()
        self.run_script(['rollback', '--cold-reboot'], control=True)
        self.assertFalse(self.package.exists())
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), STOCK)
        self.assertIn('reboot', self.actions())
        self.assertNotIn('service start', self.actions())

    def test_expired_guard_cannot_be_lost_behind_a_stuck_operation_lock(self):
        self.install(); self.boot()
        (self.root / 'tmp/opc-clockfix.lock').mkdir()
        self.assertNotEqual(self.run_script(['guard', '00000000-0000-4000-8000-000000000001'], control=True, check=False).returncode, 0)
        self.assertIn('reboot', self.actions())
        self.assertFalse((self.root / 'etc/init.d/S30opc-clockfix').exists())
        self.assertNotIn('umount', self.actions())

    def test_unrelated_mounted_binary_is_not_unmounted(self):
        self.install(); self.boot(); (self.root / 'usr/bin/prudynt').write_bytes(b'foreign repair')
        self.assertNotEqual(self.run_script(['rollback'], control=True, check=False).returncode, 0)
        self.assertNotIn('umount', self.actions())
        self.assertEqual((self.root / 'usr/bin/prudynt').read_bytes(), b'foreign repair')


if __name__ == '__main__':
    unittest.main()
