# Frequently asked questions

## How do I join?

Apply with a name, contact email and password. Submit one application per person. Sign back in with that email to check approval status; approval emails are not sent. Once approved, you can open the workspace and download the credentials for your account.

## Which credential should I use?

Use your application email and password to access your application and approved workspace. Organizer-issued participants can use their participant ID and portal credential instead. The separate gateway learning key belongs in your research client. Do not put either credential in a submission ZIP.

## What do I submit?

One ZIP containing any subset of system folders, each with `predict.py` and optional learned model files or Python helpers. It defines `predict(initial, interventions, context)`. The upload limit is 30 MiB, with at most 300 MiB of expanded contents.

## When do I do the research?

During development. Use the black-box API, Gemma and other permitted tools to collect observations, investigate mechanisms, write code and fit your finished model. Scoring runs only the uploaded numerical forecaster: no further observations, experiments, network or model calls.

## What stays the same for final scoring?

Every participant's development, public and final runs share each family's physical parameters and active mechanisms. Forecast schedules and visible initial conditions differ. Hidden state resets use the same fixed convention; there is no secret random prehistory.

## What is the budget?

Each participant has **2,000 simulator steps per system** for the research period. Steps include waiting, preparation and recovery. Reset, document, brief and budget reads are free. The allowance does not reset daily. Gemma calls do not consume simulator steps.

## How do I access Gemma?

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey), then use it to call Gemma directly. This key is separate from your simulator gateway key. The [getting-started guide](README.md) includes a Google SDK example. Keep both credentials out of your submission.

## Does Gemma have an experiment budget?

No. The simulator budget applies only to physical simulator steps. The hackathon platform imposes no Gemma token allowance or development model-call cap. Google's quotas and rate limits apply to your Google API key. Manage API access and quota in Google AI Studio.

## Is Gemma use enforced by scoring?

No. Gemma supports document interpretation, hypothesis formation, experiment design, analysis and modeling code during development. Its use earns no points by itself. Scoring measures the accuracy of your finished numerical forecaster, which cannot make LLM calls after upload.

## Can I use a coding assistant to build my model?

Yes, during development. The uploaded program must forecast using its bundled code and model files alone.

## What does `context` contain?

A plain mapping with exactly `protocol`, `revision`, `family`, `observables`, `intervention_bounds`, `brief` and `documents`. It is static across episodes. It contains no `llm` method, simulator handle, episode identity or hidden coefficients. Inputs already use physical names and units. See [PROMPT.md](PROMPT.md) for field types.

## What does the runner give my model?

For each of 40 episodes, a noisy initial physical observation and all 4,000 future physical interventions. Return 4,000 dictionaries, one after each intervention, including every observable with finite numeric values. Do not include the initial observation as the first forecast. No true intermediate outcomes are fed back.

## What can run in the sandbox?

Python 3.12 with NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 and joblib 1.5.2. The submission has 2 CPUs, 3 GiB RAM (3,072 MiB) and 1,200 seconds total across model loading and all 40 episodes. There is no network or package installation. Read bundled files relative to `__file__`. Loaded weights can persist; reset the simulated state from each call's `initial`.

## Does an upload spend my research credits?

Numerical scoring spends no physical-query or model-call credits. Upload slots are a separate allowance. Each participant/system has three accepted uploads per Toronto calendar day, shared by public and final phases combined. An accepted upload that later fails still uses its slot.

## Which upload counts?

The latest accepted public upload per system through each day counts for that day's standing; unchanged systems carry forward, rather than reverting to zero. An earlier better score does not replace the latest upload. The latest accepted final upload before close counts for final ranking. You can replace a final upload within the shared three-per-day limit while final submissions are open.

## When are the phases?

Development starts September 22, 2026. Final uploads open September 28 at 12:00 and close September 30 at 12:00, America/Toronto. Final results are held until close. Uploads accepted before close may finish afterward.

## When do I see results?

Your own public result appears when evaluation finishes.
The **public leaderboard opens September 23, 2026 at 9:00 a.m. Toronto time**, with an opening sweep scheduled for that time. Subsequent daily sweeps remain at 23:30. If selected evaluations are unfinished, publication waits until a successful sweep. Final-result timing is unchanged.

The daily public standings are swept at 23:30 Toronto time; pending selected
evaluations defer publication to a later sweep. Final standings publish at the
first successful sweep after the deadline with all selected evaluations finished,
not necessarily at noon. Your own final feedback is released at the deadline, or
when evaluation finishes if later. Until the deadline, final receipts show acceptance
only; queue status and outcome details are held. Public queue status remains visible.

## How is the leaderboard calculated?

Scoring compares forecasts with noiseless physical observables; research and initial observations contain measurement noise. Every component earns `1/(1+absolute_error/sigma)` using a frozen positive scale. Scores average over observables, steps and episodes. Sustained operation, action order, recovery and composition each contribute one quarter. Overall score averages all ten systems; missing, crashed or timed-out systems score zero. Overall score alone determines rank. Equal scores share rank; spending and tokens do not break ties.

## Is the starter a competitive solution?

It is a small numerical baseline, not a claimed score. `collect.py` saves a repeatable probe, `fit.py` fits a stable linear model to purchased records, and the starter rolls that model forward. You can replace it with equations, recurrent models, ensembles or other learned numerical forecasters.

## Can I submit several systems at once?

Yes. Upload one ZIP with a folder per included system, each containing `predict.py` and its model files. Include any subset; omitted systems keep their latest accepted model in that phase across days. Only included systems use a daily upload slot. If any included system is out of slots or missing its forecaster, the whole upload is declined without charging slots. Public and final models are separate.
