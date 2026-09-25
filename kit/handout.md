# Instalily x Google Deepmind Toronto 26 hackathon

1. Research the ten black-box systems during development. Each participant has 2,000 simulator steps per system for the research period, with no daily reset. Use Gemma directly with your API key from [Google AI Studio](https://aistudio.google.com/apikey); its usage never consumes simulator steps.
2. Fit a numerical model using your observations and permitted research tools.
3. Upload one ZIP containing any systems ready: `power_grid/predict.py`, `hospital_queue/predict.py`, and optional model files inside each system folder. Omitted systems retain their previous submission in that phase. The entrypoint is `predict(initial, interventions, context)`.
4. Forecast 40 hidden 4,000-step schedules per system. The runner supplies physical inputs and makes no model calls or new experiments.

The same physical parameters and mechanisms apply to all participants and phases. Different schedules and initial observations assess transfer of the model you researched. Four equally weighted categories cover sustained operation, action order, recovery and composition.

Three accepted uploads per participant/system/Toronto day are shared across public and final submissions. Latest public through that day counts; latest final before close counts. Final uploads open September 28 at 12:00 and close September 30 at 12:00, 2026, America/Toronto.

Overall rank is the mean across all ten system scores, with missing systems contributing zero. Equal scores share rank. See [PROMPT.md](PROMPT.md) for the complete contract and [README.md](README.md) for the executable starter.

The **public leaderboard opens September 23, 2026 at 9:00 a.m. Toronto time**, with an opening sweep scheduled for that time. Subsequent daily sweeps remain at 23:30. If selected evaluations are unfinished, publication waits until a successful sweep. Final-result timing is unchanged.

