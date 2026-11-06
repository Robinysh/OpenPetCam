# Learning from missed pets and convincing lookalikes

The detector proposes boxes; the verifier decides whether a proposed animal is your pet. Training the
verifier cannot recover a pet for which no box was proposed. The collection pool must therefore include
whole frames from ordinary footage and reported misses, alongside rejected and uncertain proposals.

`tools/hard_examples.py` builds an offline, private review queue. It reads existing images and reports;
it does not acquire footage or change the camera. Every image keeps its byte hash, source, day and
recording group. Detector scores rank review candidates; they never create ground truth. A frame without
a proposal can contain an occluded pet. An accepted proposal can be a toy.

Freeze the day assignment before reviewing model results. Keep every copy and derived crop from a
recording together. Unknown dates and conflicting duplicate provenance are quarantined. Exact duplicates
are merged; visually similar frames are down-ranked within the same scene to avoid a queue full of one
resting pose. The queue includes ordinary controls as well as difficult examples.

```bash
python3 tools/hard_examples.py mine \
  --frames eval/bootstrap/frames --reports eval/reports \
  --day-map eval/hard_examples/day_map.json \
  --out eval/hard_examples/run --limit 24

python3 tools/hard_examples_ui.py \
  --manifest eval/hard_examples/run/manifest.json \
  --queue eval/hard_examples/run/review_queue.json \
  --annotations eval/hard_examples/run/owner_annotations.jsonl --port 8788
```

Open `http://127.0.0.1:8788` on that computer. Review the small batch once:

1. Choose **Visible pet**, **No visible pet**, or **Uncertain**.
2. For a visible pet, draw a tight box around each visible part. Do not label an imagined hidden body.
3. Add an optional pose tag, then save. Navigation and page load save nothing.

The counter records how much of the batch has been reviewed. Uncertain images stay out of training.
Annotations are append-only, tied to the frozen image hash, and replacements require an explicit choice.
Test frames stay outside this review queue unless the separate sealed-review option is requested.
No visible pet means absence from that image, not absence from home.

```bash
python3 tools/hard_examples.py export \
  --manifest eval/hard_examples/run/manifest.json \
  --annotations eval/hard_examples/run/owner_annotations.jsonl \
  --out eval/hard_examples/export

python3 tools/train_detector.py train \
  --train eval/hard_examples/export/train.json \
  --dev eval/hard_examples/export/dev.json \
  --out eval/hard_examples/candidate --epochs 2 --device mps
```

The experimental trainer fine-tunes a pretrained SSDlite proposal head. It keeps candidate weights and
training membership in its output directory, chooses its threshold on development examples, and never
installs a model. Both train and development need reviewed positives and negatives. Frozen outputs are
not overwritten. The existing production detector and verifier remain separate explicit baselines.

Evaluate a frozen candidate with an independently labeled test partition and explicit baseline paths:

```bash
python3 tools/train_detector.py evaluate \
  --data eval/hard_examples/export/test.json \
  --candidate eval/hard_examples/candidate \
  --baseline-model models/yolox_nano_dyn.onnx \
  --baseline-verifier models/verifier.json \
  --out eval/hard_examples/test-result.json
```

The evaluator rejects overlap with training and development by day, group, image hash, parent frame and
identifier. Report proposal localization, recognition after the verifier, false alarms, sample and episode
counts, difficult-pose slices, and CPU latency separately. Crop accuracy is not full-frame detection recall.
Reused legacy evaluation frames are regression examples, not a fresh test. A model-assisted label is not
an owner annotation. An empty test partition cannot support an improvement claim.

Deployment requires a separate decision backed by those results and the server's runtime budget.
These commands never restart services, lower production thresholds or enable a candidate automatically.
