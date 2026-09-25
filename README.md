# Groundtruth hackathon (Instalily x Google DeepMind, Toronto 26)

Project folder: `C:\Users\ooich\groundtruth-hackathon`

## One-time setup (PowerShell)

1. Unzip so this file sits at `C:\Users\ooich\groundtruth-hackathon\README.md`.
2. Copy your credentials JSON from the portal (`app-...-credentials.json`) into `secrets\`.
3. Optional: paste your Google AI Studio key into `secrets\gemma_key.txt` (one line).
4. Install Python 3.12 from python.org if you don't have it.
5. Run:

```powershell
cd C:\Users\ooich\groundtruth-hackathon
Get-ChildItem -Recurse | Unblock-File
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Setup creates `.venv` with the exact package versions the scoring sandbox uses, then runs `check_setup.py`.

## Every new terminal

```powershell
cd C:\Users\ooich\groundtruth-hackathon
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Day one: all ten systems on the board (0 steps)

```powershell
python check_setup.py --gemma   # free: budgets, saves briefs + documents to docs\, flags any bound changes, tests Gemma key
python build_submission.py      # submission.zip, all ten systems on built-in defaults
```

Upload `submission.zip` in the Public development tab.

## Research loop, per system

```powershell
python collect_runs.py hospital_queue --dry-run pulse                      # preview, free
python collect_runs.py hospital_queue hold:recovery --steps 60             # 60 steps
python collect_runs.py hospital_queue pulse --pre 20 --dur 20 --post 40    # 80 steps
python collect_runs.py hospital_queue hold:mid --steps 60                  # 60 steps
python collect_runs.py hospital_queue step:overtime=1.0 --pre 30 --steps 90   # 90 steps, becomes the holdout
python fit.py research\hospital_queue.json --holdout 1
python build_submission.py hospital_queue
```

Upload. Repeat where the holdout proxy score is lowest.

## Files

| Path | What it is |
|---|---|
| `predict.py` | The submitted forecaster. Equations for all ten systems in one file. |
| `fit.py` | Fits one system's numbers to its research log, writes `models\<system>.json`. |
| `collect_runs.py` | Runs a planned experiment, appends it to `research\<system>.json`. Costs steps. |
| `build_submission.py` | Builds `submission.zip`, smoke-tests every folder first. |
| `check_setup.py` | Environment + model self-test + free gateway reads. Costs nothing. |
| `gateway.py` | Finds credentials (env vars, then `secrets\`). |
| `kit\` | Organizer kit and rules, untouched. |
| `research\` | Your paid experiment logs. Back this folder up. |
| `models\` | Fitted parameters, one file per system. |
| `docs\` | Briefs and documents saved by `check_setup.py`. |
| `secrets\` | Credentials. Never zipped, never committed. |

## Rules the tooling enforces for you

- Keys live only in `secrets\` or environment variables. `build_submission.py` copies only `predict.py` and `model.json`.
- Make the ZIP with `build_submission.py`. Right-click compressing the `submission` folder adds an enclosing folder, which breaks the required layout.
- `models\<system>.json` must come from our `fit.py`. A model from `kit\fit.py` is a different format and gets refused rather than silently ignored.
- `collect_runs.py` checks bounds and remaining budget before spending anything, and saves after every step.
- The kit's `predict.py` and `fit.py` are the organizer baseline. Don't mix their files with ours.
