import pandas as pd

# Load data
df_train = pd.read_csv("../data/Xy/1_train.csv")
df_test = pd.read_csv("../data/Xy/1_test.csv")

# Columns
cols_train = df_train.columns.tolist()
cols_test = df_test.columns.tolist()

# Target and features
y_train = df_train.loc[:, "label"]
X_train = df_train.drop("label", axis=1)

y_test = df_test.loc[:, "label"]
X_test = df_test.drop("label", axis=1)


# AUTO-SKLEARN

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error

# Settings
estimators_to_use = ["random_forest", "extra_trees", "gradient_boosting", "ridge_regression"]
preprocessing_to_use = ["no_preprocessing"]

# Init auto-sklearn
auto_sklearn = AutoSklearnRegressor(time_left_for_this_task=60*5,
                                    per_run_time_limit=360,
                                    include_estimators=estimators_to_use,
                                    exclude_estimators=None,
                                    include_preprocessors=preprocessing_to_use,
                                    exclude_preprocessors=None,
                                    ml_memory_limit=6156,
                                    resampling_strategy="cv",
                                    resampling_strategy_arguments={"folds": 5})

# Train models
auto_sklearn.fit(X=X_train.copy(), y=y_train.copy(), metric=mean_squared_error)
it_fits = auto_sklearn.refit(X=X_train.copy(), y=y_train.copy())

# Predict
y_hat = auto_sklearn.predict(X_test)

# Show results
auto_sklearn.cv_results_
auto_sklearn.sprint_statistics()
auto_sklearn.show_models()
auto_sklearn.get_models_with_weights()

# TPOT

from tpot import TPOTRegressor

tpot_config = {
    "sklearn.linear_model.Ridge": {},
    "sklearn.ensemble.RandomForestClassifier": {},
    "sklearn.ensemble.ExtraTreesClassifier": {},
    "sklearn.ensemble.GradientBoostingClassifier": {},
}

auto_tpot = TPOTRegressor(generations=100,
                          population_size=100,
                          offspring_size=100,
                          mutation_rate=0.9,
                          crossover_rate=0.1,
                          scoring="neg_mean_squared_error",
                          cv=5,
                          n_jobs=1,
                          max_time_mins=5,
                          verbosity=2,
                          config_dict=tpot_config)

auto_tpot.fit(features=X_train, target=y_train)

auto_tpot.fitted_pipeline_
auto_tpot.pareto_front_fitted_pipelines_
auto_tpot.evaluated_individuals_

y_hat = auto_tpot.predict(features=X_test)

# H2O AUTOML

import h2o
from h2o.automl import H2OAutoML

# Shart h2o cluster
h2o.init(max_mem_size="8G")

# Upload to h2o
df_train_h2o = h2o.H2OFrame(pd.concat([X_train, pd.DataFrame({"target": y_train})], axis=1))
df_test_h2o = h2o.H2OFrame(X_test)

features = X_train.columns.values.tolist()
target = "target"

# Training
auto_h2o = H2OAutoML(max_runtime_secs=5*60)
auto_h2o.train(x=features,
               y=target,
               training_frame=df_train_h2o)

# Leaderboard
auto_h2o.leaderboard
auto_h2o = auto_h2o.leader

# Testing
df_test_hat = auto_h2o.predict(df_test_h2o)
y_hat = h2o.as_list(df_test_hat["predict"])

# Close cluster
h2o.cluster().shutdown()

