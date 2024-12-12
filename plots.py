import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def plot_curve(dirname,
               fig=None,
               ax=None,
               title=None,
               curve="mean",
               confidence_interval=True,
               label=None,
               window_size=1,
               ):
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if title is None:
        title = f"{curve} performance"

    f_list = [os.path.join(dirname, _) for _ in os.listdir(dirname) if _.endswith('txt') and 'seed' in _]

    records = pd.DataFrame()
    for f in f_list:
        rec_ = pd.DataFrame(pd.read_csv(f, sep='\t', index_col='steps'))[[curve]]
        records = pd.concat([records, rec_], axis=1)

    records = records.dropna(axis=0)

    x_axis = records.index[window_size - 1:].to_numpy()

    mean = records[curve].mean(axis=1).to_numpy() if len(f_list) > 1 else records[curve].to_numpy()

    if window_size > 1:
        mean = [np.mean(mean[i: i + window_size]) for i in range(len(mean) - window_size + 1)]
    ax.plot(x_axis, mean, label=label if label is not None else dirname.split('_')[-1])

    if confidence_interval and curve == "mean":
        std = records[curve].std(axis=1).to_numpy()[:len(mean)] if len(f_list) > 1 else 0
        upper, lower = mean + std, mean - std
        ax.fill_between(x_axis, lower, upper, alpha=0.1)

    ax.set_xlabel("steps")
    ax.set_ylabel(f"eval {curve}")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def compare_curves(dirs):
    fig_, ax_ = plt.subplots(1, 1, figsize=(10, 6))
    for dir_name in dirs:
        fig_, ax_ = plot_curve(dir_name, label=dir_name.split('_')[-1], fig=fig_, ax=ax_)

    return fig_, ax_


def compare_hist(dirs, **kwargs):
    fig_, ax_ = plt.subplots(1, 1, figsize=(10, 6))
    for dir_name in dirs:
        fig_, ax_ = plot_curve(dir_name, label=dir_name.split('_')[-1], fig=fig_, ax=ax_, **kwargs)

    return fig_, ax_


if __name__ == '__main__':
    import os

    res_dirs = {}

    res_dirs['hopper-medium-expert-v2'] = [
        "results/hopper-medium-expert-v2/20231206-144201_dbc",
        "results/hopper-medium-expert-v2/20231130-080222_iql",
        "results/hopper-medium-expert-v2/20231130-091823_ivr",
        "results/hopper-medium-expert-v2/20231222-212024_dql",

        "results/hopper-medium-expert-v2/20231223-145343_dpiBCbasedPI",

        "results/hopper-medium-expert-v2/20231225-091603_dpi20ActorAvg"

    ]

    # hopper-medium-v2
    res_dirs['hopper-medium-v2'] = [
        # behavior cloning
        # "results/hopper-medium-v2/20231208-173915_bc",
        # "results/hopper-medium-v2/20231209-055233_dbc",

        # SOTA
        # "results/hopper-medium-v2/20231208-211552_iql",
        # "results/hopper-medium-v2/20231209-011311_ivr",
        "results/hopper-medium-v2/20231221-023639_dql",
        # "results/hopper-medium-v2/20231208-174700_dpieta0",

        # wrong coefficients!!!
        # "results/hopper-medium-v2/20231222-152937_dpiBCbaseAvgGradPI",
        # "results/hopper-medium-v2/20231222-154344_dpiTarActorAvgGradPI",
        # "results/hopper-medium-v2/20231222-200751_dpiReformlossAvgGradPI",

        # without average
        # "results/hopper-medium-v2/20231223-104353_dpiBCbasedPI",
        # "results/hopper-medium-v2/20231223-205013_dpiActorbasedPI",

        # with average
        # "results/hopper-medium-v2/20231223-104512_dpiBCbasedAvgPI",  # xt* is a re-sample from x0
        # "results/hopper-medium-v2/20231223-152932_dpiBCbasedAvgPICorrected",  # xt* \sim sqrt(1-alpha_hats) from xt
        # "results/hopper-medium-v2/20231223-161227_dpiBCbasedLocAvgPI",  # xt* \sim sqrt(1-alphas) from xt
        # "results/hopper-medium-v2/20231223-215716_dpiActorbasedAvgPI",

        # without average
        # "results/hopper-medium-v2/20231224-010911_dpiBCbased",
        "results/hopper-medium-v2/20231224-011323_dpiBCbasedeta01",
        # "results/hopper-medium-v2/20231223-205013_dpiActorbasedPI",
        "results/hopper-medium-v2/20231224-082130_dpiActorbasedeta01",
        # "results/hopper-medium-v2/20231224-112219_dpiClipActorBasedeta01",
        # "results/hopper-medium-v2/20231224-130203_dpiClipActorbasedeta01Re",  # ema -> 0.005 always update
        # "results/hopper-medium-v2/20231224-151646_dpiT10ActorbasedPI",


        # Try to average out Q-grad to find good performance!
        "results/hopper-medium-v2/20231225-011048_dpiMeanQActorbasedPI",
        "results/hopper-medium-v2/20231225-012221_dpiMeanQActorAvgPI",

        # more samples
        "results/hopper-medium-v2/20231225-063630_dpi20MeanActorbasedPI",  # use Q_min
        "results/hopper-medium-v2/20231225-091719_dpi20ActorAvg"


    ]

    res_dirs['hopper-medium-replay-v2'] = [
        "results/hopper-medium-replay-v2/20231211-100601_bc",
        "results/hopper-medium-replay-v2/20231211-104224_iql",
        "results/hopper-medium-replay-v2/20231211-114933_ivr",
        "results/hopper-medium-replay-v2/20231211-125456_dbc",
        "results/hopper-medium-replay-v2/20231221-122228_dql",

        "results/hopper-medium-replay-v2/20231211-141321_dpidbcmaxQ",
        # "results/hopper-medium-replay-v2/20231212-161513_dpieta1",

    ]
    # hopper-medium-v2  hopper-medium-replay-v2 hopper-medium-expert-v2
    data = res_dirs['hopper-medium-v2']  #
    f, a = compare_hist(data, curve='mean', window_size=1)
    f.show()
