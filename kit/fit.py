"""Fit a small stable linear state-space baseline using only purchased records."""
import argparse
import json
from pathlib import Path
import numpy as np


def fit(data, regularization=1.0):
    names = list(data['brief']['observables'])
    bounds = data['brief']['interventions']
    controls = list(bounds)
    current, actions, following = [], [], []
    for run in data['runs']:
        previous = run['initial']
        for action, observed in zip(run['actions'], run['observations']):
            current.append([previous[name] for name in names])
            following.append([observed[name] for name in names])
            actions.append([(action[name] - bounds[name][0]) / (bounds[name][1] - bounds[name][0]) for name in controls])
            previous = observed
    if not current:
        raise ValueError('Collect observations before fitting')
    current, following = np.asarray(current), np.asarray(following)
    center, scale = current.mean(axis=0), np.maximum(current.std(axis=0), 1e-6)
    x = np.column_stack(((current-center)/scale, actions, np.ones(len(current))))
    y = (following-center)/scale
    penalty = regularization * np.eye(x.shape[1]); penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    size = len(names)
    a = coefficients[:size].T
    # The starter is a stable baseline, not a discovered physical equation.
    radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if radius > 0.995:
        a *= 0.995 / radius
    return {'family': data['family'], 'observables': names, 'controls': controls,
            'bounds': bounds, 'center': center.tolist(), 'scale': scale.tolist(),
            'a': a.tolist(), 'b': coefficients[size:-1].T.tolist(),
            'bias': coefficients[-1].tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path('example_submission/model.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fit(json.loads(args.data.read_text())), indent=2)+'\n')
    print(f'Wrote numerical model to {args.output}')
