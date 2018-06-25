import pandas as pd

from autosklearn.regression import AutoSklearnRegressor
import autosklearn.metrics

import tpot
from tpot import TPOTRegressor

import h2o
from h2o.automl import H2OAutoML

class auto_ml:
    """
    Wrapper for running automatic ML models with various backends
    """

    # TODO - Add custom return for each backend

    def __init__(self, backend):
        """
        Init

        :param backend: str (sklearn, tpot, h2o) describing the backend algorithm
        """

        if backend not in ["sklearn", "tpot", "h2o"]:
            raise TypeError("Argument backend must be one of 'sklearn', 'tpot', 'h2o'")
        else:
            self.backend = backend

        self.ml_obj = None

    def create_ml(self, run_time, folds, **kwargs):
        """
        Create machine learning model using specified backend

        :param run_time: run time in minutes
        :param folds: number of cv folds
        :return: auto machine learning object
        """

        # auto-sklearn
        # https://github.com/automl/auto-sklearn
        if self.backend == "sklearn":

            # Default arguments
            default_args = {"per_run_time_limit": 360,
                            "include_estimators": ["random_forest", "extra_trees", "gradient_boosting", "ridge_regression"],
                            "exclude_estimators": None,
                            "include_preprocessors": ["no_preprocessing"],
                            "exclude_preprocessors": None,
                            "ml_memory_limit": 6156,
                            "resampling_strategy": "cv",
                            "resampling_strategy_arguments": {"folds": folds},
                            "tmp_folder": None,
                            "output_folder": None}

            # Get relevant default arguments (not in **kwargs)
            not_in_kwargs = {key: value for (key, value) in default_args.items() if key not in kwargs.keys()}

            # Expand **kwargs by defaults
            kwargs.update(not_in_kwargs)

            # Create ml object
            self.ml_obj = AutoSklearnRegressor(time_left_for_this_task=60 * run_time,
                                               per_run_time_limit=kwargs["per_run_time_limit"],
                                               include_estimators=kwargs["include_estimators"],
                                               exclude_estimators=kwargs["exclude_estimators"],
                                               include_preprocessors=kwargs["include_preprocessors"],
                                               exclude_preprocessors=kwargs["exclude_preprocessors"],
                                               ml_memory_limit=kwargs["ml_memory_limit"],
                                               resampling_strategy=kwargs["resampling_strategy"],
                                               resampling_strategy_arguments=kwargs["resampling_strategy_arguments"],
                                               tmp_folder=kwargs["tmp_folder"],
                                               output_folder=kwargs["output_folder"])

        # TPOT
        # http://epistasislab.github.io/tpot/
        elif self.backend == "tpot":

            # Models used in TPOT
            tpot_config = {
                "sklearn.linear_model.Ridge": {},
                "sklearn.ensemble.RandomForestClassifier": {},
                "sklearn.ensemble.ExtraTreesClassifier": {},
                "sklearn.ensemble.GradientBoostingClassifier": {},
            }

            # Default arguments
            default_args = {"generations": 100,
                            "population_size": 100,
                            "offspring_size": 100,
                            "mutation_rate": 0.9,
                            "crossover_rate": 0.1,
                            "scoring": "neg_mean_squared_error",
                            "max_eval_time_mins": 6,
                            "n_jobs": 1,
                            "verbosity": 0,
                            "config_dict": tpot_config,
                            "periodic_checkpoint_folder": None}

            # Get relevant default arguments (not in **kwargs)
            not_in_kwargs = {key: value for (key, value) in default_args.items() if key not in kwargs.keys()}

            # Expand **kwargs by defaults
            kwargs.update(not_in_kwargs)

            # Create ml object
            self.ml_obj = TPOTRegressor(generations=kwargs["generations"],
                                        population_size=kwargs["population_size"],
                                        offspring_size=kwargs["offspring_size"],
                                        mutation_rate=kwargs["mutation_rate"],
                                        crossover_rate=kwargs["crossover_rate"],
                                        scoring=kwargs["scoring"],
                                        cv=folds,
                                        n_jobs=kwargs["n_jobs"],
                                        max_time_mins=run_time,
                                        max_eval_time_mins=kwargs["max_eval_time_mins"],
                                        verbosity=kwargs["verbosity"],
                                        config_dict=kwargs["config_dict"],
                                        periodic_checkpoint_folder=kwargs["periodic_checkpoint_folder"])

        # H2O AutoML
        # http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
        elif self.backend == "h2o":

            # Default arguments
            default_args = {"max_models": None}

            # Get relevant default arguments (not in **kwargs)
            not_in_kwargs = {key: value for (key, value) in default_args.items() if key not in kwargs.keys()}

            # Expand **kwargs by defaults
            kwargs.update(not_in_kwargs)

            # Start H2O cluster, and clear data and create AutoML object
            h2o.init(max_mem_size="8G")
            h2o.remove_all()
            self.ml_obj = H2OAutoML(max_runtime_secs=5 * run_time,
                                    nfolds=folds,
                                    max_models=kwargs["max_models"])

        else:
            print("Unknown backend")

        return self.ml_obj

    def fit(self, X, y):
        """
        Fit method for specified backends

        :param X_train: Training data
        :param y_train: Training labels
        :return:
        """

        if isinstance(self.ml_obj, autosklearn.estimators.AutoSklearnRegressor):
            self.ml_obj.fit(X=X.copy(), y=y.copy(), metric=autosklearn.metrics.mean_squared_error)
            it_fits = self.ml_obj.refit(X=X.copy(), y=y.copy())

        elif isinstance(self.ml_obj, tpot.tpot.TPOTRegressor):
            self.ml_obj.fit(features=X, target=y)

        elif isinstance(self.ml_obj, h2o.automl.autoh2o.H2OAutoML):

            # Upload to h2o
            df_train_h2o = h2o.H2OFrame(pd.concat([X, pd.DataFrame({"target": y})], axis=1))

            # Feature and target names
            features = X.columns.values.tolist()
            target = "target"

            # Train
            self.ml_obj.train(x=features,
                              y=target,
                              training_frame=df_train_h2o)

        else:
            print("Unknown backend")

    def predict(self, X):
        """
        Predict method for specified backends

        :param X: Test data
        :return: Predicted values
        """

        if isinstance(self.ml_obj, autosklearn.estimators.AutoSklearnRegressor):
            y_hat = self.ml_obj.predict(X)

            return y_hat

        elif isinstance(self.ml_obj, tpot.tpot.TPOTRegressor):
            y_hat = self.ml_obj.predict(features=X)

            return y_hat

        elif isinstance(self.ml_obj, h2o.automl.autoh2o.H2OAutoML):
            df_test_h2o = h2o.H2OFrame(X)
            df_test_hat = self.ml_obj.predict(df_test_h2o)
            y_hat = h2o.as_list(df_test_hat["predict"])
            h2o.cluster().shutdown()

            return y_hat

        else:
            print("Unknown backend")


