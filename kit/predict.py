"""Numerical forecast baseline. Train model.json with the kit's fit.py."""
import json
from pathlib import Path
import numpy as np

_MODEL = None


def predict(initial, interventions, context):
    """Return post-intervention observations; no network or model calls."""
    global _MODEL
    path = Path(__file__).with_name('model.json')
    if not path.exists():
        # A usable zero-training baseline. Replace it with your learned model.
        return [dict(initial) for _ in interventions]
    if _MODEL is None:
        _MODEL = json.loads(path.read_text())
    model = _MODEL
    if model['family'] != context['family']:
        raise ValueError('model.json belongs to another system')
    names, controls = model['observables'], model['controls']
    center, scale = np.asarray(model['center']), np.asarray(model['scale'])
    state = (np.asarray([initial[name] for name in names]) - center) / scale
    a, b, bias = np.asarray(model['a']), np.asarray(model['b']), np.asarray(model['bias'])
    result = []
    for action in interventions:
        normalized = np.asarray([(action[name] - model['bounds'][name][0]) /
            (model['bounds'][name][1] - model['bounds'][name][0]) for name in controls])
        state = a @ state + b @ normalized + bias
        physical = np.maximum(0.0, center + scale * state)
        state = (physical - center) / scale
        result.append(dict(zip(names, physical.tolist())))
    return result
