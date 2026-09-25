# Getting started

Research the black-box systems during development, then upload a finished model. Read [the challenge](PROMPT.md) for the complete rules and [the system briefs](briefs.md) for physical inputs and outputs.

- [Participant portal](https://gt-portal-wavddee32q-uc.a.run.app/login)
- [Current kit download](https://gt-portal-wavddee32q-uc.a.run.app/participant-kit.zip)
- Research gateway: `https://gt-gateway-wavddee32q-uc.a.run.app`

New participants can [apply to participate](https://gt-portal-wavddee32q-uc.a.run.app/apply) with a contact email, name and password. Submit one application per person. Return through email sign-in to see your status; after organizer approval, enter the workspace and download your participant credentials. No approval emails are sent. Existing participants can still sign into the portal with their organizer-issued login credential. Your gateway
learning key is a separate credential; use it only with the research client.

## 1. Set up your research environment

Use Python 3.12 and install `numpy` and `httpx` in your development environment. Set `GROUNDTRUTH_GATEWAY_URL` and `GROUNDTRUTH_KEY` to the address and learning credential issued to you. Keep the key outside your files and ZIP.

From the extracted kit directory:

```sh
python -m pip install numpy httpx
export GROUNDTRUTH_GATEWAY_URL='https://gt-gateway-wavddee32q-uc.a.run.app'
export GROUNDTRUTH_KEY='your organizer-issued learning key'
```

## 2. Collect data and fit the starter model

The following commands spend **128 of your 2,000 power-grid simulator steps**. Run them once you are ready to begin experimenting.

```sh
python collect.py --system power_grid --steps 128 --output power-grid-research.json
python fit.py power-grid-research.json --output example_submission/model.json
```

The collector saves each response locally. It refuses to overwrite an earlier output file. It is a simple repeatable starting probe; change your research design as you learn. The fitter uses only those purchased records to estimate a small stable linear state-space model. The starter loads `model.json` and rolls its learned state forward through the given physical controls. Without a model file, it provides a persistence baseline.

## 3. Package and upload

From the kit directory, place your finished forecaster in a system folder:

```sh
mkdir -p models/power_grid
cp example_submission/predict.py example_submission/model.json models/power_grid/
cd models
zip -r ../submission.zip power_grid
cd ..
```

Open the participant workspace, choose the Public development tab, and upload `submission.zip`. Your receipt shows whether each model is queued, running, scored or failed. Add other system folders alongside `power_grid` when ready and include them in the same ZIP command. Each folder contains its own `predict.py`, learned files and helpers. Omitted systems keep their previous submission in the same phase. Do not include an enclosing `models/` directory, virtual environment or credentials. Maximum ZIP: **30 MiB total**, **300 MiB expanded**. A single-system ZIP with `predict.py` at the root also works when you choose its system under Single-system ZIP.

## 4. Continue your research with Gemma

Get your **Google API key for Gemma development from [Google AI Studio](https://aistudio.google.com/apikey)**. Use it directly with Google, separately from the gateway learning key. Do not put either key in code, model files or your submission ZIP.

The **2,000-step allowance per system is only for simulator experiments**, including waiting and recovery. Gemma calls do not consume it. The hackathon platform sets no Gemma token allowance or development call-count cap; Google quotas and rate limits apply to your Google API key. Manage API access and quota in Google AI Studio.

Install the Google SDK in your development environment and set `GEMMA_API_KEY` to your Google AI Studio API key. This SDK is a research dependency, not a submission dependency.

```sh
python -m pip install google-genai
export GEMMA_API_KEY='your Google AI Studio API key'
```

```python
import os
import json
from google import genai
from client import Client

with Client(os.environ['GROUNDTRUTH_GATEWAY_URL'], os.environ['GROUNDTRUTH_KEY']) as simulator:
    research = {
        "brief": simulator.brief("power_grid"),
        "documents": simulator.documents("power_grid"),
    }
    print(simulator.budget("power_grid")["simulator_steps_remaining"])

with genai.Client(api_key=os.environ['GEMMA_API_KEY']) as gemma:
    response = gemma.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=("Read these documents. Propose two competing explanations and "
                  "a short experiment to distinguish them. Treat both as hypotheses.\n"
                  + json.dumps(research)),
    )
    print(response.text)
```

Use a Gemma model enabled for your Google API key; `gemma-4-31b-it` is another configured model. The example sends only the brief and documents you explicitly include to Google, not credentials or simulator access. Add your own measurements or code when useful. You choose and run the experiments separately through the simulator API.

Use the simulator client to collect experimental observations and the Google SDK to work with Gemma during development. The uploaded forecaster runs numerically without network, simulator or model calls.

## Forecast contract

```python
def predict(initial, interventions, context):
    return [dict(initial) for _ in interventions]  # replace with your learned dynamics
```

`initial` is the first physical observation. `interventions` is the complete future physical schedule. Return every observable after every intervention, with exactly as many output dictionaries as actions. No subsequent truth is supplied. Initial and research observations are noisy; hidden scoring compares forecasts with noiseless physical observables.

`context` is a plain mapping containing `protocol`, `revision`, `family`, `observables`, `intervention_bounds`, `brief`, and `documents`. It has no methods or model API. The exact field definitions are in [the challenge](PROMPT.md). Documents and context are static across participants and episodes; hidden dynamics stay undisclosed.

One Python 3.12 process forecasts **40 × 4,000 steps** with **2 CPUs, 3 GiB RAM (3,072 MiB) and 1,200 seconds total**. NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 and joblib 1.5.2 are available. Load weights from paths relative to `__file__`; initialize trajectory state separately on every call. Scoring provides no network, simulator or LLM access.

## Submissions and standings

The physical parameters and active mechanisms are the same across participants, development, public scoring and final private scoring. Only schedules and initial observations differ. Your fitted knowledge transfers to the final phase.

The **public leaderboard opens September 23, 2026 at 9:00 a.m. Toronto time**, with an opening sweep scheduled for that time. Subsequent daily sweeps remain at 23:30. If selected evaluations are unfinished, publication waits until a successful sweep. Final-result timing is unchanged.

You have **three accepted uploads per participant/system/Toronto day across both phases combined**. The latest public upload through that day counts for daily standings; unchanged systems carry forward. Only systems included in a ZIP consume a slot. If any included system cannot be accepted, none are charged. The latest final upload before close counts for final standings. Development opens September 22, 2026; final uploads open September 28 at 12:00 and close September 30 at 12:00, America/Toronto. Final receipts show acceptance only until close; queue status and outcomes are held. Your own final feedback becomes available at close or when evaluation finishes, if later. Public standings are swept at 23:30 Toronto time. Final standings publish at the first successful sweep after close when all selected evaluations have finished; they are not promised at noon.

Overall rank is the mean score across all ten systems. Missing systems contribute zero, and equal scores share rank. There are no spending or token tie-breaks. The score covers sustained operation, action order, recovery history and control composition equally. Submit your final models explicitly in the Final private tab; public uploads do not automatically carry into the final phase.

## Before you upload

Call your forecaster locally with a sequence of valid interventions. It should return one complete, finite observation dictionary per action. Use only the published runtime dependencies, load model files relative to `__file__`, and start fresh trajectory state on every call. Keep keys, research credentials and virtual environments out of the ZIP.

Keep a copy of every model you submit. The latest accepted upload counts, even if it scores below an earlier version.
