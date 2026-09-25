# Instalily x Google Deepmind Toronto 26 hackathon

Investigate ten black-box simulators. Use experiments to discover how they behave, then build numerical models that forecast long, unfamiliar sequences of interventions.

Development opens **September 22, 2026**. Final submissions open **September 28 at 12:00** and all uploads close **September 30 at 12:00**, **America/Toronto**.

## The research challenge

This is an individual competition. Apply and submit under your own participant account.

There are ten simulated families: epidemic, market, traffic, power grid, supply chain, wildlife, reservoir, ad auction, social contagion and hospital queue. Their equations and parameters are hidden. Exactly two of three candidate mechanisms operate in each family. These are invented mathematical systems, not models suitable for real-world decisions.

Use the development black-box API, Gemma, coding assistants and other permitted research tools to design experiments, interpret observations and fit your forecaster. You can work interactively throughout development. Your submission is the finished forecaster you develop during the week.

For each system, **the same physical parameters and active mechanisms apply to every participant, development, public scoring and final private scoring**. Different runs and evaluation episodes use different initial observations and intervention schedules. There is no new participant-specific or final-phase physical world to learn. Reset returns the hidden state to its fixed public initialization convention and randomizes only the listed observable initial quantities. Observations are noisy; resets do not undo spent credits.

## Gemma’s role

**Gemma is your research assistant during development. You use it to help investigate the simulators and build your numerical forecasters.** You decide what information to give it, which experiments to run, and which model to submit.

Use Gemma to:

- **Interpret documents:** extract assumptions, operating constraints and possible mechanisms from the system briefs and operational documents.
- **Form hypotheses:** propose competing explanations for observations you have collected, including delays, depletion and interactions between controls.
- **Design experiments:** suggest interventions that could distinguish those explanations while staying within your remaining budget.
- **Analyze results:** compare experimental responses, identify where your current model fails, and decide what to investigate next.
- **Develop modeling code:** help implement equations, fit parameters, debug code and improve your finished forecaster.

For example, a power-grid model may predict a quick recovery after reserve dispatch, while your observations show a slower rebound. Give Gemma the relevant brief, intervention history and measurements. Ask it to propose explanations and an experiment that would separate them. Run that experiment through the simulator API, inspect the result, then update your numerical model. Gemma’s suggestions are hypotheses to investigate, not access to the hidden equations or an answer key.

### How to use it

Get your **Google API key from [Google AI Studio](https://aistudio.google.com/apikey)**, then use it to call Gemma directly. This is separate from your simulator gateway credential. Include the documents, measurements or code you want it to consider; Gemma does not automatically run simulator experiments for you. The [getting-started guide](README.md) includes an example.

**Gemma calls do not spend simulator credits.** The hackathon platform does not impose a Gemma token allowance or development model-call limit. Your Google API key is subject to its provider quota and rate limits. Manage your API access and quota in Google AI Studio. Gemma supports development; the submission format does not enforce its use.

### What happens after submission

**You submit the forecaster you built, not a Gemma prompt or a research agent.** Hidden evaluation calls your `predict(initial, interventions, context)` function using the code and learned files in your ZIP. It cannot call Gemma, access the network or collect new simulator observations.

Gemma use itself earns no points and is not enforced by the submission format. Your score measures forecast accuracy. The purpose of using Gemma is to improve the research and modeling decisions that lead to your finished submission.

## Simulator experiment budget

Each participant has **2,000 simulator steps per system** for the research period. Every physical step costs one credit, including waiting, preparation and recovery. Reset, document, brief and budget reads are free. The allowance does not reset daily.

Gemma calls do not consume simulator steps. Use Gemma directly with your API key from [Google AI Studio](https://aistudio.google.com/apikey).

`client.budget(system_id)` reports `simulator_steps_remaining`. Its compatibility field `remaining` is the same balance in simulator credits, at one credit per step.

## Submit your finished model

Upload **one ZIP containing any subset of systems**. Each system has a folder named exactly for its system ID, containing `predict.py` and any learned weights or helpers. Each folder is evaluated independently with the same numerical limits. Include shared helpers separately in every folder that needs them.

```text
submission.zip
├── power_grid/
│   ├── predict.py
│   └── model.json
└── hospital_queue/
    └── predict.py
```

The total ZIP limit is **30 MiB**, with at most **300 MiB expanded contents**. Use system folders directly at the ZIP root, without an enclosing submission folder. Include one, several or all ten systems. You can also upload a single-system ZIP with `predict.py` at the root by selecting its system in the portal.

Only included systems consume one of their daily upload slots. Acceptance is all-or-nothing: if an included system has no slots left or a folder is incomplete, none are accepted or charged. The receipt lists each included system. Public receipts show scoring progress; final receipts show acceptance only until the deadline. Omitted systems retain their latest accepted submission in the same phase, including across public days. A system never submitted in that phase contributes zero. Public models do not automatically become final submissions.

## Forecast inputs and outputs

The runner calls this function once per episode:

```python
def predict(initial, interventions, context):
    # initial: noisy physical observation before the first intervention
    # interventions: complete future physical action sequence, in order
    # context: the static mapping described below
    # Return one physical observation dictionary AFTER each intervention.
    ...
```

`initial` is a dictionary mapping each observable name to its noisy numeric value before the first action. `interventions` is a list of 4,000 action dictionaries, each mapping physical intervention names to numeric settings within the published bounds. The list is in time order. `forecasts` must be a list of 4,000 dictionaries with the same observable names as `initial`.

Every returned dictionary must contain every observable with finite numeric values. Return exactly one dictionary per action; do not include `initial` as a forecast step. Read optional model files relative to `__file__`. Begin the dynamical state from the supplied `initial` on each call. A process may reuse loaded model weights across episodes, but one episode's simulated state must not become another's initial state.

`context` is a plain dictionary with these exact keys:

| Key | Value |
|---|---|
| `protocol` | Internal metadata string; your forecaster can ignore it |
| `revision` | Internal metadata string; your forecaster can ignore it |
| `family` | The system name |
| `observables` | List of physical observable names |
| `intervention_bounds` | Mapping from each action name to `[low, high]` |
| `brief` | Public system description and units |
| `documents` | Public static documents, each `{id, title, text}` |

The context is the same for every participant and episode of a system. It contains no hidden parameters, episode identifier, secret interpretation or mutable host cache. The actions and observations already use physical names and units. 

## Hidden evaluation

The runner imports `predict.py` and calls `predict(initial, interventions, context)` for **40 hidden episodes, each 4,000 steps**. The full future intervention schedule is provided at once. You receive no subsequent true observations. The uploaded program makes **no Gemma or other model calls, no simulator queries and no network requests**. All research and fitting needed by your finished artifact happen before upload.

## Runtime and file limits

The runtime is Python **3.12**, with NumPy **2.3.5**, SciPy **1.16.3**, scikit-learn **1.7.2** and joblib **1.5.2**. One numerical process has **2 CPUs**, **3 GiB RAM (3,072 MiB)** and **1,200 seconds total** for loading and all 40 forecasts. Include your model files in the ZIP. Do not include credentials, a virtual environment or dependencies needing downloads.

## Evaluation scenarios

Forecast schedules stay inside the published physical action bounds. The four equally represented categories are:

1. **Sustained operation:** hold settings long enough to reveal continuing responses.
2. **Action order:** similar actions and durations appear in different orders.
3. **Recovery history:** repeated stress is separated by different recovery intervals.
4. **Composition:** controls act separately or jointly in different sequences.

## How scores are calculated

Research measurements and the initial observation contain noise. Forecasts are scored against the underlying **noiseless physical observables**, not a fresh noisy measurement.

For each observable and tick, the score is `1 / (1 + absolute_error / sigma)`. The organizer fixes each positive observable scale from frozen reference trajectories. Scores average across observables, ticks and the 40 episodes. The four categories each contribute one quarter. A higher score is better; the score is not a percentage of correctly predicted ticks.

The **overall leaderboard score is the average across all ten systems**. Missing, crashed or timed-out system submissions score zero. All ten remain in the denominator. Overall score alone determines rank; equal scores share rank. Research spending, token usage and the sequence-transfer breakdown do not break ties.

## Upload limits and deadlines

Each participant may make **three accepted uploads per system per Toronto calendar day, across public and final phases combined**. Using both tabs does not create six slots. The allowance resets at Toronto midnight. A readable ZIP with each required forecaster file can be accepted before execution; a later crash still consumes its accepted-upload slot.

For public daily standings, the **latest accepted public upload for that system up to and including that day** counts, even if an earlier upload scored better. For final standings, the **latest accepted final upload for that system before the final deadline** counts. Public and final submissions have separate opening windows, but share the daily upload allowance. An accepted upload may finish after the deadline; a replacement cannot be accepted after close.

Final scores and outcome details remain hidden until September 30 at 12:00 Toronto time. Your acceptance receipt remains available, but final queue status, scores and outcome details are withheld until that deadline. The final phase uses hidden schedules and initial conditions from the same physical systems researched during development.

The **public leaderboard opens September 23, 2026 at 9:00 a.m. Toronto time**, with an opening sweep scheduled for that time. Subsequent daily sweeps remain at 23:30. If selected evaluations are unfinished, publication waits until a successful sweep. Final-result timing is unchanged.

Your own public result appears when its asynchronous evaluation finishes. Public
daily standings are swept at **23:30 America/Toronto** using that day's latest
accepted public uploads. If a selected evaluation is still running, publication
is deferred to a later sweep rather than treating unfinished work as a zero. Final
standings publish at the first successful sweep after the final deadline when all
selected evaluations have finished. The scheduled sweep is at 23:30 Toronto time,
so final standings are not promised at the noon deadline. Your own final feedback
becomes available at the deadline, or when its evaluation finishes if later.

Follow the [getting-started guide](README.md), read the [system briefs](briefs.md), and download the participant kit for the research client and `example_submission/predict.py`. The supplied linear numerical starter is a baseline, not an organizer solution or an expected score.
