from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import random

def sim_data():

    # Parameters
    n_samples = random.randint(500, 5000)
    n_features = random.randint(5, 25)
    n_informative = random.randint(5, n_features)
    noise = random.uniform(0.5, 2)

    # Simulate data
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           noise=noise)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Param dict
    params = {"n_samples": n_samples,
              "n_features": n_features,
              "n_informative": n_informative,
              "noise": noise}

    # Return
    return X_train, y_train, X_test, y_test, params




