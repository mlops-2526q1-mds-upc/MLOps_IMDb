from contextlib import nullcontext
import importlib

import pytest

MODULE_PATHS = [
    "mlops_imdb.data.prepare",
    "mlops_imdb.features.build_features",
    "mlops_imdb.modeling.train",
    "mlops_imdb.modeling.eval",
]

NULL_CONTEXT_TYPE = type(nullcontext())


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_create_tracker_disabled_returns_nullcontext(module_path):
    module = importlib.import_module(module_path)

    tracker_ctx = module.create_tracker({}, "test-project")

    assert isinstance(tracker_ctx, NULL_CONTEXT_TYPE)


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_create_tracker_enabled_uses_emissions_tracker(tmp_path, module_path, monkeypatch):
    module = importlib.import_module(module_path)
    captured_args = {}

    class DummyTracker:
        def __init__(self, project_name, output_dir, output_file):
            captured_args["project_name"] = project_name
            captured_args["output_dir"] = output_dir
            captured_args["output_file"] = output_file
            self.final_emissions = 0.123

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(module, "EmissionsTracker", DummyTracker)

    output_path = tmp_path / "emissions" / "run.csv"
    params = {"energy": {"codecarbon": {"enabled": True, "output": str(output_path)}}}

    tracker_ctx = module.create_tracker(params, "test-project")

    assert isinstance(tracker_ctx, DummyTracker)
    assert captured_args == {
        "project_name": "test-project",
        "output_dir": str(output_path.parent),
        "output_file": output_path.name,
    }
    assert output_path.parent.exists()


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_create_tracker_enabled_defaults_to_emissions_csv(module_path, monkeypatch):
    module = importlib.import_module(module_path)
    captured_args = {}

    class DummyTracker:
        def __init__(self, project_name, output_dir, output_file):
            captured_args["project_name"] = project_name
            captured_args["output_dir"] = output_dir
            captured_args["output_file"] = output_file

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(module, "EmissionsTracker", DummyTracker)

    params = {"energy": {"codecarbon": {"enabled": True}}}

    tracker_ctx = module.create_tracker(params, "test-project")

    assert isinstance(tracker_ctx, DummyTracker)
    assert captured_args == {
        "project_name": "test-project",
        "output_dir": ".",
        "output_file": "emissions.csv",
    }
