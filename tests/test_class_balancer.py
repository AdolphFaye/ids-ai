import pytest
import pandas as pd
import numpy as np
from src.preprocessing.class_balancer import ClassBalancer

@pytest.fixture
def sample_imbalanced():
    X = pd.DataFrame(np.random.randn(1000, 5))
    y = pd.Series(['normal']*900 + ['attack']*100)
    return X, y

def test_smote_increases_minority(sample_imbalanced):
    X, y = sample_imbalanced
    balancer = ClassBalancer()
    X_bal, y_bal = balancer.fit_resample(X, y)
    assert len(y_bal) > len(y)

def test_class_weights_returns_dict(sample_imbalanced):
    X, y = sample_imbalanced
    balancer = ClassBalancer()
    weights = balancer.compute_class_weights(y)
    assert isinstance(weights, dict)
    assert len(weights) == 2

def test_strategy_none(sample_imbalanced):
    X, y = sample_imbalanced
    balancer = ClassBalancer()
    balancer.strategy = "none"
    X_bal, y_bal = balancer.fit_resample(X, y)
    assert len(y_bal) == len(y)