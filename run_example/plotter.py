import csv
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('.')


COLORS = (
    [
        '#318DE9', # blue
        '#FF7D00', # orange
        '#E52B50', # red
        '#8D6AB8', # purple
        '#00CD66', # green
        '#FFD700', # yellow
    ]
)


def merge_csv(root_dir, query_file, query_x, query_y):
    """Merge result in csv_files into a single csv file."""
    csv_files = []
    for dirname, _, files in os.walk(root_dir, followlinks=True):
        for f in files:
            if f == query_file:
                csv_files.append(os.path.join(dirname, f))
    results = {}
    for csv_file in csv_files:
        content = [[query_x, query_y]]
        df = pd.read_csv(csv_file)
        values = df[[query_x, query_y]].values
        for line in values:
            if np.isnan(line[1]): continue
            content.append(line)
        results[csv_file] = content
    assert len(results) > 0
    sorted_keys = sorted(results.keys())
    sorted_values = [results[k][1:] for k in sorted_keys]
    content = [
        [query_x, query_y+'_mean', query_y+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, query_y.replace('/', '_')+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)
    return output_path

def merge_csv_cocoa(root_dir, csv_files, query_x, query_y, min_epoch=100000):
    """Merge result in csv_files into a single csv file."""
    results = {}
    for csv_file in csv_files:
        content = [[query_x, query_y]]
        df = pd.read_csv(csv_file, usecols=range(23))
        # values = df[[query_x, query_y]].values
        values = df[[query_y]].values[1::2][:min_epoch]
        values = np.concatenate([np.array([[1000*i] for i in range(1, len(values)+1)]), values], axis=1)
        for line in values:
            if np.isnan(line[1]): continue
            content.append(line)
        results[csv_file] = content
    assert len(results) > 0
    sorted_keys = sorted(results.keys())
    sorted_values = [results[k][1:] for k in sorted_keys]
    content = [
        [query_x, query_y+'_mean', query_y+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, query_y.replace('/', '_')+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)
    return output_path

def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(
    results,
    x_label,
    y_label,
    xlim=None,
    ylim=None,
    title=None,
    smooth_radius=10,
    figsize=None,
    dpi=None,
    color_list=None,
    legend_outside=False
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if color_list == None:
        color_list = [COLORS[i] for i in range(len(results))]
    else:
        assert len(color_list) == len(results)
    for i, (algo_name, csv_file) in enumerate(results.items()):
        x, y, shaded = csv2numpy(csv_file)
        y = smooth(y, smooth_radius)
        shaded = smooth(shaded, smooth_radius)
        ax.plot(x, y, color=color_list[i], label=algo_name)
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.2)
    ax.set_title(title, fontdict={'size': 10})
    ax.set_xlabel(x_label, fontdict={'size': 10})
    ax.set_ylabel(y_label, fontdict={'size': 10})
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend_outside:
        ax.legend(loc=2, bbox_to_anchor=(1,1), prop={'size': 10})
    else:
        ax.legend(prop={'size': 10})


import matplotlib.pyplot as plt

def plot_subfigures(
    results_list,
    x_label,
    y_label,
    title_list=None,
    smooth_radius=10,
    color_list=None,
    legend_outside=False,
    figsize=None,
    dpi=None,
    nrows=1,
    ncols=1,
    fontsize=10,
    save_path=None,
):
    nrows = len(results_list) // ncols + (1 if len(results_list) % ncols != 0 else 0)
    # ncols = 4
    figsize = (5*ncols, 5*nrows) if figsize is None else figsize
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.2, hspace=0.4)

    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])  # Convert axs to a 2D array
    elif nrows == 1 or ncols == 1:
        axs = np.array(axs).reshape(nrows, ncols)  # Ensure axs is a 2D array

    if nrows == 1:
        axs = axs.reshape(1, -1)  # Ensure axs is a 2D array

    for i, results in enumerate(results_list):
        ax = axs[i//ncols, i%ncols]

        if color_list is None:
            color_list_iter = [COLORS[j] for j in range(len(results))]
        else:
            color_list_iter = color_list

        for j, (algo_name, csv_file) in enumerate(results.items()):
            x, y, shaded = csv2numpy(csv_file)
            y = smooth(y, smooth_radius)
            shaded = smooth(shaded, smooth_radius)
            ax.plot(x, y, color=color_list_iter[j], label=algo_name)
            ax.fill_between(x, y-shaded, y+shaded, color=color_list_iter[j], alpha=0.2)

        ax.set_title(title_list[i] if title_list else f"Plot {i+1}", fontdict={'size': fontsize})
        ax.set_xlabel(x_label, fontdict={'size': fontsize})
        ax.set_ylabel(y_label, fontdict={'size': fontsize})
        ax.tick_params(axis='both', which='major', labelsize=int(fontsize/3*2))
        ax.xaxis.offsetText.set_fontsize(int(fontsize//3*2))

        if legend_outside:
            ax.legend(loc=2, bbox_to_anchor=(1,1), prop={'size': fontsize})
        else:
            ax.legend(prop={'size': fontsize})

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()



def plot_func(
    root_dir,
    task,
    algos,
    query_file,
    query_x,
    query_y,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    title=None,
    smooth_radius=10,
    figsize=None,
    dpi=None,
    colors=None,
    legend_outside=False
):
    results = {}
    for algo in algos:
        path = os.path.join(root_dir, task, algo)
        csv_file = merge_csv(path, query_file, query_x, query_y)
        results[algo] = csv_file

    plt.style.use('seaborn')
    plot_figure(
        results=results,
        x_label=xlabel,
        y_label=ylabel,
        xlim=xlim,
        ylim=ylim,
        title=title,
        smooth_radius=smooth_radius,
        figsize=figsize,
        dpi=dpi,
        color_list=colors,
        legend_outside=legend_outside
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotter")
    parser.add_argument("--root_dir", default="log")
    parser.add_argument("--task", default="hopper-medium-expert-v2")
    parser.add_argument("--algos", type=str, nargs='*', default=["mobile&penalty_coef=1.5&rollout_length=5&real_ratio=0.05&auto_alpha=True"])
    parser.add_argument("--query_file", default="policy_training_progress.csv")
    parser.add_argument("--query_x", default="timestep")
    parser.add_argument("--query_y", default="eval/normalized_episode_reward")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="Timesteps")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--colors", type=str, nargs='*', default=None)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--output_path", default="./hopper-medium-expert.png")
    parser.add_argument("--figsize", type=float, nargs=2, default=(8, 6))
    parser.add_argument("--dpi", type=int, default=500)
    args = parser.parse_args()

    results = {}
    for algo in args.algos:
        path = os.path.join(args.root_dir, args.task, algo)
        csv_file = merge_csv(path, args.query_file, args.query_x, args.query_y)
        results[algo] = csv_file

    plt.style.use('seaborn')
    plot_figure(
        results=results,
        x_label=args.xlabel,
        y_label=args.ylabel,
        title=args.title,
        smooth_radius=args.smooth,
        figsize=args.figsize,
        dpi=args.dpi,
        color_list=args.colors
    )
    if args.output_path:
        plt.savefig(args.output_path)
    if args.show:
        plt.show()
