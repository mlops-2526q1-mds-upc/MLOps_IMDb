import joblib
import pandas as pd
import scipy.sparse as sp

from mlops_imdb.modeling import train


def test_main_trains_and_saves_model(tmp_path, monkeypatch):
    features_path = tmp_path / "train_features.npz"
    labels_csv = tmp_path / "train_clean.csv"
    model_path = tmp_path / "model.pkl"

    X = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    sp.save_npz(features_path, X)
    pd.DataFrame({"label": [0, 1]}).to_csv(labels_csv, index=False)

    params = {
        "data": {
            "processed": {"train": str(labels_csv)},
            "schema": {"label_col": "label"},
        },
        "features": {"outputs": {"train_features": str(features_path)}},
        "train": {
            "logreg": {"max_iter": 200, "random_state": 42},
            "outputs": {"model_path": str(model_path)},
        },
    }

    monkeypatch.setattr(train, "load_params", lambda path="params.yaml": params)
    train.main()

    assert model_path.exists()

    model = joblib.load(model_path)
    assert hasattr(model, "coef_")
    preds = model.predict(X)
    assert set(preds) <= {0, 1}
