"""Exercise software Privacy and restore the original active state. Run on the server.

This smoke test checks the real API/control contract and JPEG recovery. Transport,
recording and paused-reboot checks are separate release evidence. It does not test
physical power isolation. An already-paused camera is left untouched.
"""
import os
import sys
import time

sys.path.insert(0, '/opt/petcam')
import app


def flag_identity():
    st = os.stat(f'{app.STATE_DIR}/privacy')
    return st.st_ino, st.st_mtime_ns


def fresh_picture():
    deadline = time.monotonic() + 30
    while True:
        try:
            data = app.live_frame()
            if len(data) > 1000 and data.startswith(b'\xff\xd8'):
                return len(data)
        except app.Busy:
            raise
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise AssertionError('Fresh JPEG did not recover')
        time.sleep(1)


def main():
    if app.privacy_requested():
        print('Privacy already requested; left untouched.')
        return
    print('Before: fresh JPEG bytes', fresh_picture())
    owned = None
    try:
        response = app.set_privacy(True)
        owned = flag_identity()
        state = response['privacy_status']
        assert state['requested'] and state['confirmed'], state
        try:
            app.live_frame()
        except app.Busy:
            pass
        else:
            raise AssertionError('Paused live image request was not blocked')
        if state['mode'] == 'capture_stopped':
            app._privacy_verify_guard(app._privacy_guard('status'), True)
        else:
            assert state['mode'] == 'blackout', state
            app._privacy_verify_config(app.prudynt({'privacy': {'enabled': None}, 'audio': {'mic_enabled': None}}), True)
        print('Paused:', state['mode'], '; camera remains electrically powered.')
    finally:
        if owned is not None and app.privacy_requested() and flag_identity() == owned:
            app.set_privacy(False)
        elif app.privacy_requested():
            print('Privacy intent changed during the test; left requested.')
    if not app.privacy_requested():
        print('Restored: fresh JPEG bytes', fresh_picture())


if __name__ == '__main__':
    main()
