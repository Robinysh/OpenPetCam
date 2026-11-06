"""Native regression for the helper linked into the pinned Prudynt patch.

The full firmware additionally needs source-fragment/target-build checks and a
real SDK-boundary/RTP trial. This host test does not certify a camera binary.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools/firmware/prudynt-timestamps'


class PrudyntTimestampTest(unittest.TestCase):
    def test_native_access_unit_and_capture_clock_invariants(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if not compiler:
            self.skipTest('a host C++ compiler is required for the timestamp helper test')
        with tempfile.TemporaryDirectory(prefix='opc-clock-test-') as tmp:
            binary = Path(tmp) / 'clock-check'
            build = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    '-I', str(TOOLS), str(TOOLS / 'regression.cpp'), '-o', str(binary)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('750 pictures, 29.96s span', result.stdout)


if __name__ == '__main__':
    unittest.main()
