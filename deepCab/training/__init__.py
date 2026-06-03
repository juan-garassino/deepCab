"""Training lifecycle as pure functions: preprocess / train / evaluate / predict
/ cv / hpo / provenance. Each consumes typed configs, returns typed results, no
global state. The Hydra entry in `train.py::main` is the only side-effectful
surface; everything else can be called directly by tests and by the agent."""
from deepCab.training.cv import CVResult, run_cv  # noqa: F401
from deepCab.training.evaluate import EvalResult  # noqa: F401
from deepCab.training.hpo import HPOResult, tune  # noqa: F401
from deepCab.training.predict import predict_many, predict_one  # noqa: F401
# Note: NOT re-exporting `preprocess` here — its name collides with the
# submodule path `deepCab.training.preprocess`, which makes monkeypatching
# (and tooling that resolves dotted import strings) silently latch onto the
# function instead of the module. Callers do `from deepCab.training.preprocess
# import preprocess` explicitly.
from deepCab.training.preprocess import clean, featurize, load  # noqa: F401
from deepCab.training.provenance import Provenance, emit_provenance  # noqa: F401
from deepCab.training.seed import set_all  # noqa: F401
from deepCab.training.train import TrainResult, run  # noqa: F401
