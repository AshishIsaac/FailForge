# FailForge: automated model QA and targeted synthetic data

Your detector works in daylight and fails at night. FailForge **finds out on its own**, figures out
**what kind** of failure it is, **generates labeled training images** of exactly that situation with
Stable Diffusion + ControlNet, **retrains**, and **promotes the new model only if a frozen golden test
set proves it is better**, with confidence intervals and against the honest alternatives
(more epochs, classical augmentation, real labeled night data).

One command sets everything up and opens a dashboard where you start the loop and watch every stage.

![dashboard](docs/images/dashboard_results.png)

<!--RESULTS-->

---

## Contents

1. [Quick start](#1-quick-start)
2. [What happens on the first run](#2-what-happens-on-the-first-run)
3. [Using the dashboard](#3-using-the-dashboard)
4. [The experiment](#4-the-experiment)
5. [How it works](#5-how-it-works)
6. [Command line](#6-command-line)
7. [CI/CD and model validation](#7-cicd-and-model-validation)
8. [Docker](#8-docker)
9. [Performance](#9-performance)
10. [Repository layout](#10-repository-layout)
11. [Updating, moving and uninstalling](#11-updating-moving-and-uninstalling)
12. [Troubleshooting](#12-troubleshooting)
13. [License and data](#13-license-and-data)

---

## 1. Quick start

### Step 1: check you have what's needed

| | Requirement |
|---|---|
| **OS** | Windows 10/11 (64-bit) or Linux (Ubuntu 22.04 / 24.04). macOS works in CPU mode. |
| **Python** | **3.10, 3.11 or 3.12, 64-bit.** Newer versions (3.13+) are not supported yet. |
| **GPU** | An NVIDIA GPU with **6 GB or more** and a recent driver (560+) is strongly recommended. You do **not** need to install CUDA; FailForge installs what it needs. Without one everything runs on the CPU, but training and image generation become very slow (use the `smoke` profile to try it). |
| **RAM** | 16 GB. |
| **Disk** | About **12 GB** free, **on an internal drive** (not a USB stick): PyTorch with CUDA ~4 GB, BDD100K 0.7 GB, models 4.4 GB, plus runs. |
| **Network** | The first run downloads about 8 GB (once). |

**Installing Python (Windows).** Open <https://www.python.org/downloads/release/python-31210/>,
download **Windows installer (64-bit)**, run it, tick **"Add python.exe to PATH"**, click
**Install Now**. Already have a newer Python? Install 3.12 anyway, it sits next to it;
`start.bat` finds it by itself.

**Linux:** `sudo apt install git python3-venv python3-tk`.

### Step 2: get the code

```bash
git clone https://github.com/AshishIsaac/FailForge.git
cd FailForge
```

### Step 3: run it

| Windows | Linux / macOS |
|---|---|
| double-click **`start.bat`** (or `python run.py`) | `./start.sh` (or `python3 run.py`) |

The **first time**, a setup window appears and installs and downloads everything (20–60 minutes,
depending on your connection); then your browser opens the dashboard. **Every later time**, the same
command opens the dashboard straight away.

### Step 4: start a loop

In the dashboard, pick a profile and click **Start a new loop**:

| profile | what it does | time on a 6 GB laptop GPU |
|---|---|---|
| **Smoke test** | 40 images, 1 epoch, classical synthesis: checks that every stage works | 5 min |
| **Quick** | 1,500 day images, 8 + 4 epochs, 150 ControlNet images | about 1 h |
| **Full (reference)** | all 4,218 day images, 30 + 12 epochs, 500 ControlNet images: the experiment in this README | about 5 h |

Before you run anything, the dashboard already shows the **reference run** (the full loop on the
author's machine, committed in `reports/reference/`), so you can explore every view straight away.

Useful options (all remembered where it makes sense):

```bash
python run.py --loop quick          # open the dashboard and start a quick loop immediately
python run.py --home D:\FailForge   # keep data, models and runs (~8 GB) on another drive (remembered)
python run.py --venv D:\ffenv       # put the Python environment elsewhere (remembered)
python run.py --cpu                 # force CPU PyTorch
python run.py --reinstall           # rebuild the environment from scratch
python run.py --port 9000 --no-browser
```

## 2. What happens on the first run

`run.py` uses only the Python standard library. It creates a private environment (`.venv`) and
runs these steps, showing each one in the setup window (details in `.setup/install.log`):

1. **Python environment**, then **PyTorch**: the CUDA 12.6 build if `nvidia-smi` finds an NVIDIA GPU
   (it bundles CUDA and cuDNN), otherwise the CPU build.
2. **FailForge and its libraries** (Ultralytics, diffusers, transformers, scikit-learn, UMAP, FastAPI…),
   with upper version bounds so a future release cannot break a fresh clone.
3. **BDD100K**: 10,000 real dashcam images with labels and time-of-day / weather tags (700 MB).
4. **Splits and the frozen golden set** (`failforge data prepare`): see [docs/data.md](docs/data.md).
5. **Models**: YOLO11n (detector) and YOLO11s (QC teacher), CLIP ViT-B/32, Stable Diffusion 1.5,
   ControlNet canny + depth, Depth Anything V2 Small (about 4.4 GB in total).
6. **Verify** (`failforge doctor`), then the dashboard starts.

Every completed step is recorded: if the setup is interrupted (network, closed window), running
the same command again **resumes** where it stopped. Changing `pyproject.toml` or `requirements/`
re-runs only what changed. Everything downloaded goes into one folder, `workspace/` (or your
`--home`), so uninstalling is deleting it.

## 3. Using the dashboard

* **Runs** (left): the reference run and every loop on this machine, with their state. Click one to view it.
* **Pipeline**: the ten stages with live progress (epochs, images generated, acceptance rate), and
  why the loop ran (the drift test and SLO check). **Stop** stops cleanly at the next checkpoint;
  **Resume** continues from the last finished stage (a stopped generation resumes mid-way).
* **Failure modes**: an interactive UMAP map of every mined error (hover a point; click a cluster
  card to highlight it), the TIDE error breakdown, each failure mode's error rate, lift and
  examples, and its synthesis target. Toggle *colour by true time of day* to check that the
  unsupervised clustering really found "night" on its own.
* **Synthesis**: quality-control statistics, a KID comparison (which data looks most like real
  night), and examples (source photo → Canny / depth controls → generated night image with the
  inherited boxes), including one the filters rejected.
* **Results & gate**: golden-set mAP for every slice and arm, the confidence-interval plot, and the
  promotion-gate checklist. **Open full report** opens the self-contained HTML report.

A loop runs as its own process: closing the browser, or even the dashboard, does not stop it.

## 4. The experiment

The plan was to make the failure happen on purpose, so the fix can be measured:

1. **Baseline**: YOLO11n (COCO-pretrained) fine-tuned on **daytime BDD100K images only**, the way a
   team that collected data in the easy conditions would.
2. **Golden set**: 1,102 frozen images, 302 day, 600 night and 200 dawn/dusk, stratified by weather.
   Every model is scored on exactly these images (COCO mAP@50:95), overall and per slice.
3. **The loop** finds the failures on a separate labeled "production probe" (mostly night), clusters
   them, synthesizes night images for the top failure modes, and fine-tunes.
4. **Ablation**: four arms, all fine-tuned from the same baseline for the same 12 epochs on the same
   day images plus the same number *N* of extra images:

| arm | extra data | question it answers |
|---|---|---|
| more epochs | none | is any gain just extra training? |
| classical augmentation | N day images darkened, tinted, noised, with light blooms | is the cheap trick enough? |
| **ControlNet synthetic** | N generated night images (passed QC) | **the method** |
| real night | N real labeled night/dawn images | the upper bound, if we had paid for labels |

5. **Gate**: the synthetic model is promoted only if the night slice gains ≥ 2 mAP points with a
   paired-bootstrap 95% CI above zero, overall mAP drops by at most 0.5 points, and no other golden
   slice drops by more than 2 points.

<!--RESULTS_TABLE-->

## 5. How it works

The full design, with C4 diagrams, data contracts and design decisions, is in
[docs/architecture.md](docs/architecture.md). In short:

* **Monitoring.** MMD on CLIP embeddings with a permutation test (training images vs the production
  window), a domain-classifier AUC, brightness PSI, and per-slice SLOs on the golden set. Either
  one triggers the loop; otherwise the run stops with a health report.
* **Failure mining.** Every prediction above the deployment threshold is typed TIDE-style: missed,
  background false positive, localization, class confusion, both, duplicate.
* **Failure modes.** Each error, plus a sample of true positives (so every cluster has an error
  *rate*), becomes a CLIP embedding of a context crop plus photometric descriptors; UMAP and
  HDBSCAN find the clusters; CLIP zero-shot enrichment names them ("night / dark object");
  clusters are ranked by **excess errors**. The real tags are used only to validate the result.
* **Synthesis.** The image budget is split across failure modes by excess errors. For each mode,
  labeled day photos rich in its classes are re-rendered by **Stable Diffusion 1.5 img2img with two
  ControlNets** (Canny edges + Depth Anything depth) under the target condition. img2img starts
  from a classically relit copy so night really becomes night, while ControlNet pins the
  structure from the original, so **the source boxes stay valid**. That is then checked:
  * CLIP prompt adherence, and P(target condition) from CLIP zero-shot ("did it become night?");
  * **label survival**: an independent COCO detector (YOLO11s) must still find the labeled objects
    where it found them in the source photo.
* **Metrics.** COCO mAP implemented in numpy and verified against pycocotools to 1e-6. Image slices
  and bootstrap resamples are just weight vectors over images, which makes paired-bootstrap
  confidence intervals for every arm and slice cheap.
* **Registry.** Promoted models become the champion (`workspace/registry/<profile>/champion.json`);
  the next loop starts from the champion.

## 6. Command line

Everything the dashboard does is available from the CLI (inside the environment:
`.venv\Scripts\failforge` on Windows, `.venv/bin/failforge` on Linux):

```bash
failforge loop -c quick                       # run a loop (profiles: smoke, quick, full, or a YAML path)
failforge loop -c full --resume               # resume the latest full run
failforge loop -c quick --set synthesis.count=300 --set gate.min_target_gain=0.01
failforge loop -c full --run-id X --only gate,report   # re-run single stages (others come from cache)
failforge eval --weights reports/reference/weights/synthetic.pt   # score any checkpoint on the golden set
failforge gate --check reports/reference      # re-derive the promotion decision from saved artifacts
failforge report workspace/runs/<id>          # rebuild a report
failforge bench --synth                       # latency / throughput / memory benchmarks
failforge data validate --check-files         # schema, leakage and golden-lock checks
failforge registry --profile full             # champion and promotion history
failforge doctor                              # what is installed / downloaded
```

Every option is documented in [`src/failforge/config.py`](src/failforge/config.py); unknown keys are
an error, so a typo never silently falls back to a default.

## 7. CI/CD and model validation

Free GitHub runners have no GPU, so the work is split honestly:

* **`ci.yml`** (every push / PR): ruff; unit, API and launcher tests; the **whole loop with a fake
  detector and fake CLIP** (stages, caching, resume, stop, registry, gate, report); a **real CPU
  smoke loop** on a 400-image BDD100K subset (real YOLO training, CLIP, UMAP/HDBSCAN, synthesis,
  bootstrap, gate; the report is uploaded as an artifact); **re-deriving the promotion decision
  from the committed reference artifacts** (`failforge gate --check reports/reference --expect promote`);
  Docker build + health check.
* **`model-validation.yml`** (manual, weekly, or when detector / data / synthesis code changes):
  the real loop with ControlNet on a **self-hosted GPU runner**; the job fails if the candidate does
  not pass the gate.

## 8. Docker

```bash
docker compose up                                  # dashboard on http://localhost:8765 (NVIDIA GPU)
docker compose run --rm failforge loop -c quick    # a loop without the dashboard
docker build -f docker/Dockerfile --build-arg TORCH=cpu -t failforge-cpu .   # CPU image
```

The first start downloads the data and models into the `failforge-ws` volume. The GPU image needs
the NVIDIA driver and the NVIDIA Container Toolkit on the host.

## 9. Performance

<!--BENCH-->

Full numbers and how to reproduce them: [docs/benchmarks.md](docs/benchmarks.md)
(`python benchmarks/run_all.py`).

## 10. Repository layout

```
run.py, start.bat, start.sh     one-command launcher (stdlib only)
configs/                        profiles: full (reference), quick, smoke, ci
src/failforge/
  data/        BDD100K download, manifests, stratified group-aware splits, golden lock, YOLO store
  eval/        COCO matching, weighted mAP + paired bootstrap, TIDE error typing, slices
  mining/      failure mining on the production probe
  analysis/    CLIP embeddings, photometrics, UMAP/HDBSCAN, failure-mode naming, drift tests
  synthesis/   prompts, ControlNet generator, classical relighting, quality filters, engine
  loop/        staged resumable pipeline, promotion gate, model registry
  report/      HTML/Markdown report, publishing the reference run
  bench/       latency / throughput / memory benchmarks
  app/         dashboard (FastAPI + vanilla JS)
reports/reference/              committed results + checkpoints of the full reference run
docs/                           architecture (C4), data, benchmarks, images
tests/                          45 tests, incl. the full loop with fakes and a pycocotools cross-check
docker/, docker-compose.yml     container
.github/workflows/              ci.yml (every PR), model-validation.yml (GPU runner)
dvc.yaml                        the same stages for DVC users (optional)
```

## 11. Updating, moving and uninstalling

* **Update:** `git pull`, then run as usual; only changed dependencies are installed.
* **Move the workspace:** `python run.py --home NEW_DIR` (move the old `workspace/` there first to
  keep your data and runs).
* **Uninstall:** delete the folder (and your `--home` / `--venv` folders, if you used them).

## 12. Troubleshooting

| problem | fix |
|---|---|
| "Python 3.10 – 3.12 was not found" | install Python 3.12 (step 1); `start.bat` offers to do it with winget |
| setup failed | read `.setup/install.log`, fix the cause (often the network), run again: it resumes |
| the setup takes hours | you are probably on a USB stick or SD card: use `--home` and `--venv` on an internal drive |
| `MemoryError` in a DataLoader worker | close other programs; or `--set detector.workers=1` (workers are already capped by RAM) |
| CUDA out of memory during synthesis | keep `synthesis.cpu_offload=true` (default); close other GPU programs |
| slow HF downloads / rate limits | set `HF_TOKEN` (a free Hugging Face token) before running |
| a loop was interrupted | select it in the dashboard and click **Resume** (or `failforge loop -c <profile> --resume`) |
| the golden set refuses to load | the data changed since it was frozen; see docs/data.md "The golden set is frozen" |

## 13. License and data

Code: MIT (see [LICENSE](LICENSE)). **BDD100K** is licensed for non-commercial research and
education only; FailForge downloads it at first run and never redistributes images. The committed
checkpoints are fine-tuned from Ultralytics YOLO11n (AGPL-3.0) on BDD100K and inherit both terms.
Stable Diffusion 1.5 (CreativeML OpenRAIL-M), ControlNet v1.1, CLIP (MIT) and Depth Anything V2
Small (Apache-2.0) are downloaded from their publishers.
