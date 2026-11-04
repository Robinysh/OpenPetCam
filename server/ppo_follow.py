"""Follow by a learned recurrent policy (PPO, trained in the calibrated simulator: sim/train_ppo.py, sim/env.py).

The network (models/follow_ppo.onnx, ~100 KB: 64-unit LSTM) runs with onnxruntime once per watcher cycle on the
observation from server/follow_obs.py, the same code that built its training observations. It picks one of:
  0 wait · 1 centre on the latest sighting · 2 lead-aim (sighting + velocity x (age + 1 s)) · 3/4 turn 0.75 frame
  left/right · 5/6 tilt 0.6 frame up/down · 7 search (the tracker's own routine) · 8 stop
The hard rules are enforced here, never left to the network: no motor move while she was just seen lying still, and at
most MOVE_BUDGET moves per BUDGET_WINDOW (follow_obs.py). Its second output (fast detection) is logged, not acted on.
"""
import numpy as np

from follow_obs import N_OBS, FollowObs

LSTM_HIDDEN = 64
N_MOTOR = 9


class CountingCam:
    """The tracker's camera API, counting every move it starts (so a search's moves count against the budget)."""

    def __init__(self, cam, clock, on_move):
        self._cam, self._clock, self._on_move = cam, clock, on_move

    def __getattr__(self, name):
        return getattr(self._cam, name)

    def goto(self, x, y):
        self._on_move(self._clock())
        return self._cam.goto(x, y)

    def aim(self, fx, fy):
        self._on_move(self._clock())
        return self._cam.aim(fx, fy)


class PPOFollow:
    def __init__(self, model_path, cam, tracker_cls, find_pet, clock, log, sleep, find_now=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.clock, self.log = clock, log
        self.F = FollowObs()
        self.cam = CountingCam(cam, clock, self.F.note_move)
        self.tracker = tracker_cls(self.cam, find_pet, clock=clock, log=log, sleep=sleep, find_now=find_now)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, LSTM_HIDDEN), np.float32)
        self.c = np.zeros((1, 1, LSTM_HIDDEN), np.float32)

    @property
    def searching(self):
        return self.tracker.searching

    def decide(self, obs):
        logits, self.h, self.c = self.sess.run(None, {"obs": obs[None], "h": self.h, "c": self.c})
        lg = logits[0]
        return int(np.argmax(lg[:N_MOTOR])), int(np.argmax(lg[N_MOTOR:]))

    def step(self, t, fresh, box, stale_box, motion, weak_n, pose, busy):
        """One watcher cycle. box: this cycle's fresh verified box or None; stale_box: what the cycle shows (may be an
        older detector result, as in training). Returns an action string (or None) like the tracker's."""
        self.F.update(t, fresh, box, motion, weak_n)
        self.F.box = stale_box
        obs = self.F.vector(t, pose[0], pose[1], busy)
        assert obs.shape == (N_OBS,)
        motor, fast = self.decide(obs)
        if fast:
            self.log("ppo: asked for fast detection (not enabled on this server)")
        if motor == 0:
            return None
        if motor == 8:
            self.cam.stop()
            return "ppo: stop"
        why = self.F.blocked(t, motor)
        if why:
            return None
        last = self.F.sightings[-1] if self.F.sightings else None
        n0 = len(self.F.move_times)
        act = None
        if motor in (1, 2):
            if not last or t - last[0] > 3.0:
                return None
            vx, vy = self.F.velocity(t) if motor == 2 else (0.0, 0.0)
            lag = t - last[0] + 1.0
            fx, fy = last[1] + vx * lag, last[2] + vy * lag
            self.cam.aim(fx, fy)
            act = f"track: ppo {'lead-aim' if motor == 2 else 'centre'} ({fx:+.2f}, {fy:+.2f})"
        elif motor in (3, 4):
            self.cam.aim(-0.75 if motor == 3 else 0.75, 0.0)
            act = f"chase: ppo turn {'left' if motor == 3 else 'right'}"
        elif motor in (5, 6):
            self.cam.aim(0.0, -0.6 if motor == 5 else 0.6)
            act = f"chase: ppo tilt {'up' if motor == 5 else 'down'}"
        elif motor == 7:
            act = self.tracker.search("ppo") or "search (ppo): done"
        if len(self.F.move_times) > n0:
            self.F.last_move_t = self.clock()          # as trained: counted from when the action finished
        return act
