import time
import json

from load_data import load_data
from auto_ml import auto_ml

from sklearn.metrics import mean_squared_error

# Files
files = [{"train": "../data/01_r_air_train.csv", "test": "../data/01_r_air_test.csv", "task": "regression"},
         {"train": "../data/02_r_bike_train.csv", "test": "../data/02_r_bike_test.csv", "task": "regression"},
         {"train": "../data/03_r_gas_train.csv", "test": "../data/03_r_gas_test.csv", "task": "regression"}]

# Backends
backends = ["sklearn", "tpot", "h2o"]

# Settings
time_to_run = 30  # in minutes
folds = 5  # for cv

# Result container
results = []

# Loop over datasets
for data in files:

    # Load data
    X_train, y_train, X_test, y_test = load_data(path_train=data["train"], path_test=data["test"])

    # Loop over backends
    for engine in backends:

        # Start time tracking
        start_time = time.time()

        # Init model
        mod = auto_ml(backend=engine)
        mod.create_ml(run_time=time_to_run, folds=folds)

        # Fitting on training set
        mod.fit(X=X_train, y=y_train)

        # Predict on test set
        y_hat = mod.predict(X=X_test)

        # End time tracking
        run_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        # Eval error on test set
        mse_score = mean_squared_error(y_true=y_test, y_pred=y_hat)

        # Results
        info = {"train": data["train"],
                "test": data["test"],
                "task": data["task"],
                "backend": engine,
                "mse_test": mse_score,
                "run_time": run_time}
        results.append(info)

# Save model results to json
with open("../models/" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time())) + ".json", "w") as outfile:
    json.dump(results, outfile, sort_keys=True, indent=4)

