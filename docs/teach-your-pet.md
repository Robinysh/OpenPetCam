# Teach it your pet

```bash
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy pillow
.venv/bin/python tools/bootstrap_pet.py all      # collect crops, label them with `claude -p`, train, report
.venv/bin/python tools/bootstrap_pet.py deploy   # after reading the report
```

Out of the box the verifier asks "is this any dog breed?". The labeller is told about your pet and its lookalikes
(`PET_DESCRIPTION`, `PET_LOOKALIKES`); labels are cached, so a re-run only labels new crops. A trained verifier
replaces the default only once there are enough labels across at least 3 days to test it honestly on held-out
days, and `./deploy.sh` refuses to replace a trained verifier with the default. The 👍/👎 answers in the app are the
human labels used to check it. Method and numbers: [docs/evaluation.md](evaluation.md).

To label by hand, `python3 tools/label_ui.py` opens a local page: the full frame with an editable box and one key
per label, likely mistakes and night crops first. To check Find, `python3 tools/search_review.py pull` copies what
every search saw (each stop, each sighting, with scores); label them in the page's "Searches" queue, then
`python3 tools/search_review.py report` gives how often "found her" was really her, by day and night.
