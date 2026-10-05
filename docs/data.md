# Data: BDD100K, splits and the frozen golden set

## Source

FailForge uses **BDD100K** (Berkeley DeepDrive): real 1280×720 dashcam frames from US cities, with
2-D boxes and, crucially, per-image **time of day**, **weather** and **scene** tags. The first run
downloads the Hugging Face mirror [`dgural/bdd100k`](https://huggingface.co/datasets/dgural/bdd100k):
the 10,000 images of the official validation split with labels in FiftyOne format, about 700 MB,
no account needed.

| tag | values in the 10,000 images |
|---|---|
| time of day | day 5,258 · night 3,929 · dawn/dusk 778 · undefined 35 |
| weather | clear 5,346 · overcast 1,239 · undefined 1,157 · snowy 769 · rainy 738 · partly cloudy 738 · foggy 13 |
| scene | city street 6,112 · highway 2,499 · residential 1,253 · other 136 |

**License.** BDD100K is licensed for non-commercial research and education
(https://doc.bdd100k.com/license.html). FailForge never redistributes images; the checkpoints in
`reports/reference/weights/` were trained on it and inherit those terms.

## Classes

BDD labels are mapped to 8 detection classes (`src/failforge/data/bdd.py`); `train`, `trailer` and
`other vehicle` (≈100 boxes in total) are dropped, as are boxes thinner than 4 px (label noise).

| FailForge class | BDD labels |
|---|---|
| pedestrian | pedestrian, other person |
| rider | rider |
| car | car |
| truck | truck |
| bus | bus |
| two_wheeler | bicycle, motorcycle |
| traffic_light | traffic light |
| traffic_sign | traffic sign |

## Splits

`failforge data prepare` builds five disjoint splits (numbers from the reference workspace):

| split | images | boxes | composition | used by |
|---|---:|---:|---|---|
| `golden` | 1,102 | 19,953 | day 302 · night 600 · dawn 200, stratified by weather | every evaluation, the gate |
| `probe` | 501 | 8,853 | day 101 · night 300 · dawn 100 | failure mining, drift monitoring (never trained on) |
| `val_day` | 400 | 7,971 | day | validation curves during training |
| `train_day` | 4,218 | 85,358 | day | the baseline and every arm |
| `target_pool` | 2,971 | 49,145 | night 2,692 · dawn 279 | only the *real* (upper-bound) arm, and the KID reference |

* **Day-only training is deliberate.** It recreates the most common real-world failure (a
  detector trained on the conditions that were easy to collect) with a known cause, so the loop's
  effect is measurable.
* **Stratification** is proportional within each (time of day, weather) cell, so the golden
  night slice contains rainy and snowy nights in their natural proportion.
* **Group-aware assignment.** Images are assigned by hashing the BDD video id (the part of the
  file name before the dash), so frames that could come from the same drive never straddle two
  splits. `failforge data validate` re-checks this, plus label sanity (classes, degenerate or
  out-of-image boxes, duplicate ids).

## The golden set is frozen

On the first `prepare`, `data/manifests/golden.lock.json` records the golden image ids and an
order-independent SHA-256 digest of their labels. Every later `prepare` rebuilds the golden set
*from the lock*, not from the split parameters, and refuses to run if an image is missing or a
label changed: every model ever scored by this workspace was scored on exactly the same images and
labels. Changing the golden-set size in a config later has no effect, by design. To start over
deliberately, delete the lock (all previous scores become incomparable).

## Manifest format (bring your own data)

Every set in FailForge is a JSON-lines manifest, one image per line:

```json
{"id": "b1c66a42-6f7d68ca", "image": "C:/…/b1c66a42-6f7d68ca.jpg", "width": 1280, "height": 720,
 "boxes": [{"cls": 2, "x1": 100.0, "y1": 210.5, "x2": 380.0, "y2": 330.0, "occluded": false, "truncated": false}],
 "attrs": {"timeofday": "night", "weather": "rainy", "scene": "city_street", "source": "real"}}
```

To use another dataset, write `data/manifests/all.jsonl` in this format (boxes in absolute
pixels, classes indexed into `CLASSES` in `config.py`) and run `failforge data prepare`. The
`timeofday` / `weather` attributes drive the slices (`src/failforge/eval/slices.py`); other
attributes can be added as new slices with a one-line predicate.
