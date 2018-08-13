import json, os, fnmatch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

