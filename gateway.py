"""
gateway.py  -  one place that knows where credentials live.

Research scripts call open_client(). Credentials come from, in order:
  1. GROUNDTRUTH_GATEWAY_URL / GROUNDTRUTH_KEY environment variables
  2. the organizer's credentials JSON dropped into secrets/ (any *credentials*.json)
Gemma key: GEMMA_API_KEY env var, else secrets/gemma_key.txt.

Nothing here prints a key. secrets/ never goes into a submission ZIP
(build_submission.py only copies predict.py and model.json).

The kit's client.py is loaded by file path on purpose: kit/ also contains
predict.py and fit.py, and putting kit/ on sys.path would let
`import predict` pick up the kit's baseline instead of ours.
"""
import glob
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS = os.path.join(HERE, "secrets")


def credentials():
    """Return (gateway_url, gateway_key), or (None, None) if not configured."""
    url = os.environ.get("GROUNDTRUTH_GATEWAY_URL")
    key = os.environ.get("GROUNDTRUTH_KEY")
    if url and key:
        return url, key
    for path in sorted(glob.glob(os.path.join(SECRETS, "*credentials*.json"))):
        with open(path) as f:
            c = json.load(f)
        if c.get("gateway_url") and c.get("gateway_key"):
            return c["gateway_url"], c["gateway_key"]
    return None, None


def gemma_key():
    key = os.environ.get("GEMMA_API_KEY")
    path = os.path.join(SECRETS, "gemma_key.txt")
    if not key and os.path.exists(path):
        with open(path) as f:
            key = f.read().strip()
    return key or None


def _kit_client_class():
    spec = importlib.util.spec_from_file_location("kit_client", os.path.join(HERE, "kit", "client.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Client


def open_client(transport=None):
    """The kit's Client, ready to use as a context manager."""
    url, key = credentials()
    if not url:
        raise SystemExit("No gateway credentials. Copy your credentials JSON into secrets/ "
                         "or set GROUNDTRUTH_GATEWAY_URL and GROUNDTRUTH_KEY.")
    return _kit_client_class()(url, key, transport=transport)
