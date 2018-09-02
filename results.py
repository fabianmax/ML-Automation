import json, os, fnmatch
import h2o

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

from write_read_pickle import MacOSFile, pickle_load


# Get files names of results
files = fnmatch.filter(os.listdir('../results'), '*.json')

# Container
results_df = pd.DataFrame()

# Load results and add to Data Frame
for file in files:

    with open('../results/' + file) as path:
        data = json.load(path)

        results_df = results_df.append(data, ignore_index=True)

# Sort and reset index
results_df = results_df.sort_values(by=['run', 'backend']).reset_index(drop=True)

# Calculate average benchmark
results_df['mse_benchmark_avg'] = results_df.groupby(by='run')['mse_benchmark'].transform(lambda x: x.mean())

# Name
results_df['simulation'] = results_df['run'].map(lambda x: 'Simulation ' + str(int(x + 1)))

# Results without tpopt
results_df_reduced = results_df[results_df["backend"] != "tpot"]

# Function for plotting horizontal line in sns.FacetGrid
def plot_hline(y, z, **kwargs):
    data = kwargs.pop("data")
    data = data.drop_duplicates([z])
    yval = data[y].iloc[0]
    plt.axhline(y=yval, c='red', linestyle='dashed', zorder=-1)

# Plot Errors by bars
plot = sns.FacetGrid(results_df, col='simulation', sharey=False, col_wrap=5)
plot = plot.map(sns.barplot, 'backend', 'mse_test')
plot = plot.map_dataframe(plot_hline, y='mse_benchmark', z='simulation')
plot = plot.set_titles("{col_name}")
plot = plot.set_axis_labels("Algorithm", "Mean Squard Error")
plt.show()

# Plot Errors by bars (without tpot)
plot = sns.FacetGrid(results_df_reduced, col='simulation', sharey=False, col_wrap=5)
plot = plot.map(sns.barplot, 'backend', 'mse_test')
plot = plot.map_dataframe(plot_hline, y='mse_benchmark', z='simulation')
plot = plot.set_titles("{col_name}")
plot = plot.set_axis_labels("Algorithm", "Mean Squard Error")
plt.show()










# DEEP DIVE


# Load fitted autsklearn model
mod_fitted_sklearn = pickle_load('../models/2018-08-13_21-13-32_sklearn.pickle')
mod_fitted_sklearn.get_models_with_weights()
mod_fitted_sklearn.show_models()


# Load fitted h2o model
h2o.init(max_mem_size="8G", nthreads=1)
h2o.remove_all()
mod_fitted_h2o = h2o.load_model('../models/StackedEnsemble_BestOfFamily_0_AutoML_20180814_024704')


# Load/Sim data
X_train, y_train, X_test, y_test = load_data(path_train="../data/Xy/" + str(1) + "_train.csv",
                                             path_test="../data/Xy/" + str(1) + "_test.csv")

# Load predictions
p_sklearn = pickle_load("../predictions/2018-08-13_21-13-32_sklearn.pickle")
p_h2o = pickle_load("../predictions/2018-08-14_00-46-53_h2o.pickle")

# Add actuals
df_p = pd.DataFrame({"actual": y_test,
                     "p_h2o": p_h2o["predict"],
                     "p_sklearn": p_sklearn})

# Add features
df_p = pd.concat([df_p, X_test], axis=1)

predictions = ["actual", "p_h2o", "p_sklearn"]
features =[i for i in df_p.columns.values.tolist() if i not in predictions]

df_test = pd.melt(df_p, id_vars=predictions, value_vars=features, var_name="feature_name", value_name="feature_value")
df_test = pd.melt(df_test, id_vars=["feature_name", "feature_value"], var_name="pred_name", value_name="pred_value")

# Scatter by features
ax = sns.FacetGrid(df_test, col='feature_name', hue="pred_name", sharey=False, col_wrap=4)
ax = ax.map(sns.scatterplot, "feature_value", "pred_value")

# Fit by features
ax = sns.lmplot(x="feature_value", y="pred_value", hue="pred_name", col="feature_name",
                col_wrap=4, lowess=True, data=df_test, legend_out=True, height=3,
                scatter_kws={'alpha': 0.5, 'edgecolors': 'white'})
ax = ax.set_titles("{col_name}")
ax = ax.set_axis_labels("Feature Value", "Predicted Value")
ax._legend.set_title("Type")
for t, l in zip(ax._legend.texts, ['Actual', 'Prediction H2O', 'Prediction sklearn']):
    t.set_text(l)


