# Detector evaluation on the reference install

The reference install watches the pet, the author's golden retriever. These are the numbers behind the
defaults (thresholds, verifier) that ship with OpenPetCam; re-check them on your own camera with
`tools/evaluate.py` and `tools/bootstrap_pet.py`.

## Detector: evidence so far

Detector: YOLOX-nano (Apache-2.0) on the camera's 640x360 JPEG snapshot, any COCO animal
class except "teddy bear". "the pet is here" at animal score >= 0.30 (`watch.py PRESENT_SCORE`).
Tool: `tools/evaluate.py` (pull → numbered contact sheets with no scores → hand labels written
first → score → report with a dev/test split by calendar hour).

## Night 1 (2026-11-01 22:57 → 2026-11-02 03:32), 276 per-minute frames

| | frames | with the pet (hand label) | detector >= 0.30 | max score |
|---|---|---|---|---|
| all | 276 | 0 | 0 | 0.000 |

Labels: every frame inspected on 10 contact sheets before scoring. the pet was not in view all
night. Scenes: living room in IR with her ball, rope toy and rug (22:57-00:46), then the camera
tipped/sagged to face a wall at close range (00:47 →), including day/night flapping frames
(dark, noisy, magenta).

**Result: 0 false alarms, max score 0.000.** The 0.30 threshold has a wide margin on this
apartment's empty scenes, toys included.

**Not measured: recall.** With no the pet frames tonight, how often the detector *misses* her is
unknown beyond the first spot-check (3 frames of the pet from 2026-11-01, IR and upside-down,
scored 0.34-0.78, all >= 0.30; 7 empty frames all 0.00). Three positives is far too few to
trust. Single annotator, so no inter-annotator agreement yet.

## Next

Collect a day with the pet in view (`evaluate.py pull`, then label), aim for >= 100 frames with her
across day and IR, have a second person label a 50-frame subsample for
agreement, then pick the threshold on the dev half and report on the test half. If recall is
poor at 0.30 (likely failure: curled-up sleeping dog in IR, partial views), the cheapest
fix is lowering the threshold, given 0 false alarms at any score above 0 tonight.

## Sounds (2026-11-02 morning): loudness rule vs YAMNet

The loudness rule ("≥ 2 short bursts = barking") produced ~40 "barking" moments between 06:32 and
07:37 as the household woke up. YAMNet over the 26 that had clips:

| YAMNet dog score | clips | what YAMNet heard |
|---|---|---|
| ≥ 0.55 | 6 | Dog / Animal / Domestic animals (07:19-07:35, around the pet's first sighting at 07:23) |
| 0.40-0.55 | 4 | Speech, Squeak, Finger snapping, Silence/Breathing (borderline) |
| < 0.40 | 16 | Taps and sinks, dishes, frying, finger snaps, music, speech, crumpling |

**Caveat: YAMNet is a second model, not ground truth.** Nobody has listened to these clips yet.
Her sounds score high on "Squeak", "Bird", "Chicken": likely yelps/whines rather than deep barks.
The threshold 0.55 is provisional. The app now asks "Was this the pet? 👍/👎" on every sound moment;
those answers (`events.label`) are the labels to fit it on (dev/test split by day). Borderline
example to label first: 07:21:01 "Squeak", dog 0.41, peak -12 dBFS.

Applied: 6 events kept as dog sounds, 5 relabelled "loud sound" with YAMNet's label, 15 quiet
household sounds deleted with their clips, 16 events whose audio was no longer available marked
"Sound (not checked)".

## Toys vs the pet (2026-11-02 morning) and the verifier

YOLOX over 488 morning frames: 51 animal candidates; by eye 8 were the pet, 43 her black plush toy
(detected every minute from 07:26). `claude -p` labelled the same 51-crop sheet identically to my
reading (I am also a Claude model, so this is consistency, not independent confirmation).

After de-duplication, 16 unique crops (11 the pet incl. one IR night frame, 5 toy), labelled by `claude -p`:

| scorer | separates toy from the pet? |
|---|---|
| YOLOX score | no: toy up to 0.435, the pet down to 0.10 |
| zero-shot MobileNetV2 P(any dog breed) | yes on all 16: toy <= 0.294, the pet >= 0.393 |
| trained logistic regression | not evaluable: the hour-split test set held 1 crop |

Deployed: zero-shot at 0.34 (midpoint). Re-checked on frames: 07:30 toy-only -> rejected (0.09);
08:03 and 08:12 the pet -> accepted (0.96-1.00), toy beside her at 08:12 rejected (0.04).
Today's per-minute presence recomputed with it: the pet-present minutes 51 -> 9.

**Still owed:** a held-out test across days (needs more labelled days; the watcher now saves
candidate frames), and human labels as the independent check (the pet moments' 👍/👎 in the app).
