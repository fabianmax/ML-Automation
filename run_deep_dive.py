import time
import pickle
import h2o

from load_data import load_data
from auto_ml import auto_ml
from write_read_pickle import MacOSFile, pickle_dump

from sklearn.metrics import mean_squared_error

# Files
files = [{"train": "../data/Xy/" + str(1) + "_train.csv",
          "test": "../data/Xy/" + str(1) + "_test.csv",
          "task": "regression",
          "name": str(1)}]

# Backends
backends = ["sklearn", "h2o"]

# Settings
time_to_run = 60*3  # run time for each dataset and engine in minutes
folds = 5  # number of folds used in cv

# Load/Sim data
X_train, y_train, X_test, y_test = load_data(path_train=files[0]["train"], path_test=files[0]["test"])

# Loop over backends
for engine in backends:

    # Start time tracking
    start_time = time.time()

    try:

        path_model = "../models/" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time())) + "_" + str(engine) + ".pickle"
        path_pred = "../predictions/" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time())) + "_" + str(engine) + ".pickle"


        # Init model
        mod = auto_ml(backend=engine)
        mod.create_ml(run_time=time_to_run, folds=folds)

        # Fitting on training set
        mod_fitted = mod.fit(X=X_train, y=y_train)

        # Save fitted model
        if engine == "sklearn":
            model_path = pickle_dump(mod_fitted, path_model)

        elif engine == "h2o":
            model_path = h2o.save_model(model=mod_fitted.leader, path="../models/", force=True)

        # Predict on test set
        y_hat = mod.predict(X=X_test)
        pred_path = pickle_dump(y_hat, path_pred)

        # End time tracking
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        # Eval error on test set
        mse_score = mean_squared_error(y_true=y_test, y_pred=y_hat)

    except (RuntimeError, TypeError, NameError):
        print("Error")





