"""Run the watcher's SoundModel on saved clip audio (from /tmp/snd, made by the eval pull)."""
import sys
import numpy as np
sys.path.insert(0, "/opt/petcam")
import watch

m = watch.SoundModel()
idx = {line.split("|")[2]: line.split("|")[0] for line in open("/tmp/snd/index.txt")}
for t in sys.argv[1:]:
    eid = idx[t]
    pcm = np.fromfile(f"/tmp/snd/{eid}_barking.raw", dtype=np.int16)[: 16000 * 14]
    y = m.classify(pcm)
    verdict = y["dog_kind"] if y["dog"] >= watch.DOG_SOUND_THRESHOLD else f"not a dog ({y['top']})"
    print(f"{t}: dog {y['dog']:.2f} -> {verdict}")
