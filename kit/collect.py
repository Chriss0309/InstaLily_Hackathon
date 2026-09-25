"""Purchase a short repeatable physical probe, preserving each received observation."""
import argparse
import json
import os
from pathlib import Path
import random
from client import Client


def collect(client, family, steps, output, seed=17, block=16):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output path to preserve earlier research')
    if steps < 1 or block < 1:
        raise ValueError('steps and block must be positive')
    available = client.budget(family)['remaining']
    if steps > available:
        raise ValueError(f'This probe needs {steps} simulator steps; {available} remain')
    brief = client.brief(family)
    reset = client.reset(family)
    run = {'initial': reset['observation'], 'actions': [], 'observations': []}
    data = {'family': family, 'brief': brief, 'runs': [run]}
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data)+'\n')
    for tick in range(steps):
        if tick % block == 0:
            action = {name: rng.uniform(*bounds) for name, bounds in brief['interventions'].items()}
        observed = client.step(reset['run_id'], action)['observation']
        run['actions'].append(dict(action)); run['observations'].append(observed)
        output.write_text(json.dumps(data)+'\n')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, default=Path('research.json'))
    args = parser.parse_args()
    with Client(os.environ['GROUNDTRUTH_GATEWAY_URL'], os.environ['GROUNDTRUTH_KEY']) as client:
        collect(client, args.system, args.steps, args.output, args.seed, args.block)
    print(f'Saved {args.steps} purchased observations to {args.output}')
