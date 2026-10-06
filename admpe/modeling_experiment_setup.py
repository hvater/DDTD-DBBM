from admpe.excitation_experiment_setup import SUPPORTED_SYSTEMS

SUPPORTED_MODELS = ["neural_euler"]


def fetch_parameterization():
    raise NotImplementedError()


def load_dataset():
    raise NotADirectoryError()


def setup_experiment():
    raise NotImplementedError()
