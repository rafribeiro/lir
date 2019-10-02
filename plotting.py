from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def plot_data(points_h1, points_h2):
    plt.style.use('ggplot')
    plt.figure()
    plt.scatter(points_h1[:, 0], points_h1[:, 1], c='red')
    plt.scatter(points_h2[:, 0], points_h2[:, 1], c='blue')


def plot_log_lrs(uncalibrated_log_lrs, calibrated_log_lrs, train_log_lrs, params, points, train_points, fig_prefix=''):
    plt.style.use('ggplot')
    n_methods = len(uncalibrated_log_lrs) - 2
    fig, ax = plt.subplots(n_methods, 5, figsize=(12, 15))
    fig.canvas.set_window_title('dim {}, N {}'.format(params[0], params[1]))
    i = 0
    for method, ll in uncalibrated_log_lrs.items():
        if method not in ('Oracle', 'random'):
            plot_true_vs_estimated_log_lrs(ax, i, 3, method, uncalibrated_log_lrs)
            plot_true_vs_estimated_log_lrs(ax, i, 4, method, calibrated_log_lrs)
            plot_true_vs_estimated_log_lrs(ax, i, 2, method, train_log_lrs)
            for j in [3, 4, 2]:
                max_min = max(ax[i, j].get_ylim()[0], ax[i, j].get_xlim()[0])
                min_max = min(ax[i, j].get_ylim()[1], ax[i, j].get_xlim()[1])
                ax[i, j].plot([max_min, min_max], [max_min, min_max])

            cmap = plt.get_cmap('bwr')
            (vmin, vmax) = (np.min(train_log_lrs[method]), np.max(train_log_lrs[method]))
            # make symmetric around 0
            (vmin, vmax) = (-np.max([-vmin, vmax]), np.max([-vmin, vmax]))
            ax[i, 1].scatter(points[0][:, 0], points[0][:, 1], c=uncalibrated_log_lrs[method][0], edgecolors='r',
                             cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i, 1].scatter(points[1][:, 0], points[1][:, 1], c=uncalibrated_log_lrs[method][1], edgecolors='b',
                             cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i, 0].scatter(train_points[0][:, 0], train_points[0][:, 1], c=train_log_lrs[method][0],
                             edgecolors='r', cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i, 0].scatter(train_points[1][:, 0], train_points[1][:, 1], c=train_log_lrs[method][1],
                             edgecolors='b', cmap=cmap, vmin=vmin, vmax=vmax)
            ax[i, 0].set_ylabel(method)
            i += 1
    ax[i - 1, 3].set_xlabel('uncalibrated')
    ax[i - 1, 4].set_xlabel('calibrated')
    ax[i - 1, 2].set_xlabel('training')
    ax[i - 1, 1].set_xlabel('eval first 2 dim')
    ax[i - 1, 0].set_xlabel('training first 2 dim')
    plt.savefig(
        'results/{prefix} dim {dim}, N {n}'.format(prefix=fig_prefix, dim=params[0], n=params[1]).replace('.', 'x'))


def plot_true_vs_estimated_log_lrs(ax, i, j, method, log_lrs):
    ax[i, j].scatter(log_lrs['Oracle'][0], log_lrs[method][0], c='red')
    ax[i, j].scatter(log_lrs['Oracle'][1], log_lrs[method][1], c='blue')


def plot_performance(scores, fig_prefix):
    plt.style.use('ggplot')
    # refactor: for each method, for each sample size, give us a dict of dimension to scores
    refactored_scores = OrderedDict()
    for k, i in scores.items():
        for method, item in i.items():
            refactored_scores.setdefault(method, {}).setdefault(k[1], {})[k[0]] = item

    n_methods = len(refactored_scores.keys()) - 2

    fig, ax = plt.subplots(n_methods, 5, figsize=(18, 15))

    i = 0
    for method, item in refactored_scores.items():
        if method not in ('Oracle', 'random'):
            for n_sample, s in item.items():
                n_dims = list(s.keys())
                n_dims.sort()
                cllrs = [s[n_dim]['Cllr'] for n_dim in n_dims]
                cllrs_cal = [s[n_dim]['Cllr_calibrated'] for n_dim in n_dims]
                diffs = [s[n_dim]['mean log LR absolute difference (norm)'] for n_dim in n_dims]
                diffs_cal = [s[n_dim]['mean log LR absolute difference (norm)_calibrated'] for n_dim in n_dims]
                time_training = [s[n_dim]['training time'] for n_dim in n_dims]
                time_evaluation = [s[n_dim]['evaluation time'] for n_dim in n_dims]
                ax[i, 0].plot(n_dims, cllrs, label=n_sample)
                ax[i, 1].plot(n_dims, diffs, label=n_sample)
                ax[i, 2].plot(n_dims, cllrs_cal, label=n_sample)
                ax[i, 3].plot(n_dims, diffs_cal, label=n_sample)
                ax[i, 4].plot(n_dims, time_evaluation, label='eval {}'.format(n_sample))
                ax[i, 4].plot(n_dims, time_training, '--', label='train {}'.format(n_sample))
                ax[i, 0].set_ylabel(method)
                for j in [0, 1, 2, 3]:
                    ax[i, j].set_xlabel('n dimensions')
                    ax[i, j].set_ylim([0, 2])
            i += 1
    ax[i - 1, 0].set_xlabel('Cllr')
    ax[i - 1, 1].set_xlabel('absolute log LR diff (normalised)')
    ax[i - 1, 2].set_xlabel('calibrated Cllr')
    ax[i - 1, 3].set_xlabel('calibrated abs log LR diff (norm)')
    ax[i - 1, 4].set_xlabel('wall times')
    ax[i - 1, 3].legend()
    ax[i - 1, 4].legend()
    plt.savefig('results/{prefix} scores'.format(prefix=fig_prefix).replace('.', 'x'))
