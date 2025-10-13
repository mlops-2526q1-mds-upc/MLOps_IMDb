import joblib
import pandas as pd
import scipy.sparse as sp

from mlops_imdb.features import build_features


def test_main_builds_tfidf_features(tmp_path, monkeypatch):
    train_csv = tmp_path / "train_clean.csv"
    test_csv = tmp_path / "test_clean.csv"
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"

    pd.DataFrame({"text": ["good movie", "bad film"]}).to_csv(train_csv, index=False)
    pd.DataFrame({"text": ["great acting", "poor script"]}).to_csv(test_csv, index=False)

    train_features = features_dir / "train_features.npz"
    test_features = features_dir / "test_features.npz"
    vectorizer_path = models_dir / "tfidf.pkl"

    params = {
        "features": {
            "tfidf": {
                "max_features": 100,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_df": 1.0,
            },
            "outputs": {
                "train_features": str(train_features),
                "test_features": str(test_features),
                "vectorizer_path": str(vectorizer_path),
            },
        },
        "data": {
            "processed": {
                "train": str(train_csv),
                "test": str(test_csv),
            }
        },
    }

    monkeypatch.setattr(build_features, "load_params", lambda path="params.yaml": params)
    build_features.main()

    assert train_features.exists()
    assert test_features.exists()
    assert vectorizer_path.exists()

    X_train = sp.load_npz(train_features)
    X_test = sp.load_npz(test_features)
    vectorizer = joblib.load(vectorizer_path)

    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2
    assert vectorizer.max_features == 100
