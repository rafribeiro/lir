from contextlib import contextmanager
from functools import partial
import logging

import matplotlib.pyplot as plt
import numpy as np

from .bayeserror import plot_nbe as nbe
from .calibration import IsotonicCalibrator
from .ece import plot_ece as ece
from . import util


LOG = logging.getLogger(__name__)


# make matplotlib.pyplot behave more like axes objects
plt.set_xlabel = plt.xlabel
plt.set_ylabel = plt.ylabel
plt.set_xlim = plt.xlim
plt.set_ylim = plt.ylim


class Canvas:
    def __init__(self, ax):
        self.ax = ax

        self.calibrator_fit = partial(calibrator_fit, ax=ax)
        self.ece = partial(ece, ax=ax)
        self.lr_histogram = partial(lr_histogram, ax=ax)
        self.nbe = partial(nbe, ax=ax)
        self.pav = partial(pav, ax=ax)
        self.score_distribution = partial(score_distribution, ax=ax)
        self.tippett = partial(tippett, ax=ax)

    def __getattr__(self, attr):
        return getattr(self.ax, attr)


def savefig(path):
    """
    Creates a plotting context, write plot when closed.

    Example
    -------
    ```py
    with savefig(filename) as ax:
        ax.pav(lrs, y)
    ```

    A call to `savefig(path)` is identical to `axes(savefig=path)`.

    Parameters
    ----------
    path : str
        write a PNG image to this path
    """
    return axes(savefig=path)


def show():
    """
    Creates a plotting context, show plot when closed.

    Example
    -------
    ```py
    with show() as ax:
        ax.pav(lrs, y)
    ```

    A call to `show()` is identical to `axes(show=True)`.
    """
    return axes(show=True)


@contextmanager
def axes(savefig=None, show=None):
    """
    Creates a plotting context.

    Example
    -------
    ```py
    with axes() as ax:
        ax.pav(lrs, y)
    ```
    """
    fig = plt.figure()
    try:
        yield Canvas(ax=plt)
    finally:
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        plt.close(fig)


def pav(lrs, y, add_misleading=0, show_scatter=True, ax=plt):
    """
    Generates a plot of pre- versus post-calibrated LRs using Pool Adjacent
    Violators (PAV).

    Parameters
    ----------
    lrs : numpy array of floats
        Likelihood ratios before PAV transform
    y : numpy array
        Labels corresponding to lrs (0 for Hd and 1 for Hp)
    add_misleading : int
        number of misleading evidence points to add on both sides (default: `0`)
    show_scatter : boolean
        If True, show individual LRs (default: `True`)
    ax : pyplot axes object
        defaults to `matplotlib.pyplot`
    ----------
    """
    pav = IsotonicCalibrator(add_misleading=add_misleading)
    pav_lrs = pav.fit_transform(lrs, y)

    with np.errstate(divide='ignore'):
        llrs = np.log10(lrs)
        pav_llrs = np.log10(pav_lrs)

    xrange = yrange = [llrs[llrs != -np.Inf].min() - .5, llrs[llrs != np.Inf].max() + .5]

    # plot line through origin
    ax.plot(xrange, yrange)

    # line pre pav llrs x and post pav llrs y
    line_x = np.arange(*xrange, .01)
    with np.errstate(divide='ignore'):
        line_y = np.log10(pav.transform(10 ** line_x))

    # filter nan values, happens when values are out of bound (x_values out of training domain for pav)
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
    line_x, line_y = line_x[~np.isnan(line_y)], line_y[~np.isnan(line_y)]

    # some values of line_y go beyond the yrange which is problematic when there are infinite values
    mask_out_of_range = np.logical_and(line_y >= yrange[0], line_y <= yrange[1])
    ax.plot(line_x[mask_out_of_range], line_y[mask_out_of_range])

    # add points for infinite values
    if np.logical_or(np.isinf(pav_llrs), np.isinf(llrs)).any():

        def adjust_ticks_labels_and_range(neg_inf, pos_inf, axis_range):
            ticks = np.linspace(axis_range[0], axis_range[1], 6).tolist()
            tick_labels = [str(round(tick, 1)) for tick in ticks]
            step_size = ticks[2] - ticks[1]

            axis_range = [axis_range[0] - (step_size * neg_inf),axis_range[1] + (step_size * pos_inf)]
            ticks = [axis_range[0]] * neg_inf + ticks + [axis_range[1]] * pos_inf
            tick_labels = ['-∞'] * neg_inf + tick_labels + ['+∞'] * pos_inf

            return axis_range, ticks, tick_labels

        def replace_values_out_of_range(values, min_range, max_range):
            # create margin for point so no overlap with axis line
            margin = (max_range - min_range) / 60
            return np.concatenate([np.where(np.isneginf(values), min_range + margin, values),
                                   np.where(np.isposinf(values), max_range - margin, values)])

        yrange, ticks_y, tick_labels_y = adjust_ticks_labels_and_range(np.isneginf(pav_llrs).any(),
                                                                       np.isposinf(pav_llrs).any(),
                                                                       yrange)
        xrange, ticks_x, tick_labels_x = adjust_ticks_labels_and_range(np.isneginf(llrs).any(),
                                                                       np.isposinf(llrs).any(),
                                                                       xrange)

        mask_not_inf = np.logical_or(np.isinf(llrs), np.isinf(pav_llrs))
        x_inf = replace_values_out_of_range(llrs[mask_not_inf], xrange[0], xrange[1])
        y_inf = replace_values_out_of_range(pav_llrs[mask_not_inf], yrange[0], yrange[1])

        ax.yticks(ticks_y, tick_labels_y)
        ax.xticks(ticks_x, tick_labels_x)

        ax.scatter(x_inf,
                    y_inf, facecolors='none', edgecolors='#1f77b4', linestyle=':', c=y_inf)

    ax.axis(xrange + yrange)
    # pre-/post-calibrated lr fit

    if show_scatter:
        ax.scatter(llrs, pav_llrs, c=y)  # scatter plot of measured lrs

    ax.set_xlabel("pre-calibrated 10log LR")
    ax.set_ylabel("post-calibrated 10log LR")


def lr_histogram(lrs, y, bins=20, ax=plt):
    """
    plots the 10log lrs
    """
    log_lrs = np.log10(lrs)

    bins = np.histogram_bin_edges(log_lrs, bins=bins)
    points0, points1 = util.Xy_to_Xn(log_lrs, y)
    ax.hist(points0, bins=bins, alpha=.25, density=True)
    ax.hist(points1, bins=bins, alpha=.25, density=True)
    ax.set_xlabel('10log likelihood ratio')
    ax.set_ylabel('count')


def tippett(lrs, y, ax=plt):
    """
    plots the 10log lrs
    """
    log_lrs = np.log10(lrs)

    xplot = np.linspace(np.min(log_lrs), np.max(log_lrs), 100)
    lr_0, lr_1 = util.Xy_to_Xn(log_lrs, y)
    perc0 = (sum(i >= xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i >= xplot for i in lr_1) / len(lr_1)) * 100

    ax.plot(xplot, perc1, color='b', label='LRs given $\mathregular{H_1}$')
    ax.plot(xplot, perc0, color='r', label='LRs given $\mathregular{H_2}$')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlabel('10log likelihood ratio')
    ax.set_ylabel('Cumulative proportion')
    ax.legend()


def score_distribution(scores, y, bins=20, ax=plt):
    """
    plots the distributions of scores calculated by the (fitted) lr_system
    """
    ax.rcParams.update({'font.size': 15})
    bins = np.histogram_bin_edges(scores[np.isfinite(scores)], bins=bins)

    # create weights vector so y-axis is between 0-1
    scores_by_class = [scores[y == cls] for cls in np.unique(y)]
    weights = [np.ones_like(data) / len(data) for data in scores_by_class]

    # adjust weights so largest value is 1
    for i, s in enumerate(scores_by_class):
        hist, _ = np.histogram(s, bins=np.r_[-np.inf, bins, np.inf], weights=weights[i])
        weights[i] = weights[i] * (1 / hist.max())

    # handle inf values
    if np.isinf(scores).any():
        prop_cycle = ax.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x_range = np.linspace(min(bins), max(bins), 6).tolist()
        labels = [str(round(tick, 1)) for tick in x_range]
        step_size = x_range[2] - x_range[1]
        bar_width = step_size / 4
        plot_args_inf = []

        if np.isneginf(scores).any():
            x_range = [x_range[0] - step_size] + x_range
            labels = ['-∞'] + labels
            for i, s in enumerate(scores_by_class):
                if np.isneginf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[0] + bar_width if i else x_range[0], np.sum(weights[i][np.isneginf(s)])))

        if np.isposinf(scores).any():
            x_range = x_range + [x_range[-1] + step_size]
            labels.append('∞')
            for i, s in enumerate(scores_by_class):
                if np.isposinf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[-1] - bar_width if i else x_range[-1], np.sum(weights[i][np.isposinf(s)])))

        ax.xticks(x_range, labels)

        for color, x_coord, y_coord in plot_args_inf:
            ax.bar(x_coord, y_coord, width=bar_width, color=color, alpha=0.25, hatch='/')

    for cls, weight in zip(np.unique(y), weights):
        ax.hist(scores[y == cls], bins=bins, alpha=.25,
                 label=f'class {cls}', weights=weight)


def calibrator_fit(calibrator, score_range=(0, 1), resolution=100, ax=plt):
    """
    plots the fitted score distributions/score-to-posterior map
    (Note - for ELUBbounder calibrator is the firststepcalibrator)

    TODO: plot multiple calibrators at once
    """
    ax.rcParams.update({'font.size': 15})

    x = np.linspace(score_range[0], score_range[1], resolution)
    calibrator.transform(x)

    ax.plot(x, calibrator.p1, label='fit class 1')
    ax.plot(x, calibrator.p0, label='fit class 0')
