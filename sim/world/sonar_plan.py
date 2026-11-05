"""A sound-scan plan that does not confound direction with time (cybersweat lesson: points collected as consecutive
blocks make 'same direction' and 'same time' indistinguishable). Units are shuffled with a recorded seed; the only
contiguous block is the still-repeat block, which is the point of it.
    python3 -m sim.world.sonar_plan OUT.json [--seed N]
Plan entries: [kind, pan, tilt, volume]. kind: scan | remeasure | repeat_moved | repeat_still | silence | level."""
import json
import random
import sys

LEVEL, DOWN, UP = 699, 520, 880
REF = (1850, LEVEL)
PANS = [1850, 2210, 1490, 2570, 1130, 2930, 770, 3290, 410, 3650, 50]


def make(seed):
    rnd = random.Random(seed)
    units = [[("scan", p, UP, 40)] for p in PANS]
    for p, t in rnd.sample([(p, t) for p in PANS for t in (LEVEL, DOWN)], 6):
        units.append([("remeasure", p, t, 40)])
    units += [[("repeat_moved", *REF, 40)] for _ in range(5)]
    units += [[("silence", *REF, 0)] for _ in range(2)]
    units.append([("level", *REF, 30)])
    units.append([("repeat_still", *REF, 40)] * 5)
    rnd.shuffle(units)
    plan = [e for u in units for e in u]
    # a moved repeat must actually be reached from elsewhere: never directly after another reference entry
    for k in range(1, len(plan)):
        if plan[k][0] == "repeat_moved" and (plan[k - 1][1], plan[k - 1][2]) == REF:
            j = next((j for j in range(k + 1, len(plan)) if (plan[j][1], plan[j][2]) != REF), None)
            if j is not None:
                plan[k], plan[j] = plan[j], plan[k]
    return plan


if __name__ == "__main__":
    a = sys.argv[1:]
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else random.SystemRandom().randrange(1 << 30)
    plan = make(seed)
    json.dump({"seed": seed, "plan": plan, "design": __doc__.strip().splitlines()[0]}, open(a[0], "w"), indent=1)
    print(len(plan), "entries, seed", seed)
