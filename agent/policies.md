# Router policies (semantic, slow path). Editable; the code fast path is in tools/superagent.py.

- Owner reports go straight to a report worker (fast path, no router decision needed).
- A goal miss: dispatch an "improve" worker only if no open thread already works on that goal and the miss is not
  explained by something already in progress (recall the thread). Otherwise note it.
- Dispatch at most 2 workers at a time and stay within the daily budget shown in the prompt.
- Ask the owners (reply with needs_owner) only for: money, hardware, anything that moves the camera more often or
  plays sounds, privacy, deleting data. Everything else: decide.
- Replies to the owners: at most 100 words, plain words, only when something changed for them or a decision is needed.
