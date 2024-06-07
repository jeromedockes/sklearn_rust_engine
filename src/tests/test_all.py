import pytest
import sklearn_rust_engine


def test_sum_as_string():
    assert sklearn_rust_engine.sum_as_string(1, 1) == "2"
