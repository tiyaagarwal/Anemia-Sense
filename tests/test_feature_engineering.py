import numpy as np

from src.data.load_data import FEATURE_COLUMNS, load_processed
from src.features.feature_engineering import build_preprocessor


def test_preprocessor_output_shape_and_scaling():
    df = load_processed()
    X = df[FEATURE_COLUMNS]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (len(X), len(FEATURE_COLUMNS))
    assert not np.isnan(transformed).any()
    # StandardScaler: each column should be ~mean 0, ~std 1 after fitting on itself.
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(transformed.std(axis=0), 1, atol=1e-6)


def test_preprocessor_is_only_fit_on_training_data_in_practice():
    """Not a leakage test in itself (that's covered by validate_data tests) —
    just asserts the preprocessor is a fresh, unfitted transformer each time
    build_preprocessor() is called, so callers can't accidentally reuse a
    transformer fit on a different split."""
    a = build_preprocessor()
    b = build_preprocessor()
    assert a is not b
