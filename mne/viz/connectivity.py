"""Functions to plot directed connectivity
"""
from __future__ import print_function

# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: Simplified BSD


from itertools import cycle
from functools import partial

import numpy as np

from ..externals.six import string_types

from .circle import plot_connectivity_circle


def _plot_connectivity_matrix_nodename(x, y, con, node_names):
    x = int(round(x) - 2)
    y = int(round(y) - 2)
    if x < 0 or y < 0 or x >= len(node_names) or y >= len(node_names):
        return ''
    return '{} --> {}: {:.3g}'.format(node_names[x], node_names[y],
                                  con[y + 2, x + 2])


def plot_connectivity_matrix(con, node_names, indices=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, fig=None, subplot=111,
                             show_names=True):
    """Visualize connectivity as a matrix.

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    show_names : bool
        Enable or disable display of node names in the plot. The names are
        always displayed in the status bar when hovering over them.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt

    n_nodes = len(node_names)

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices must be provided if con.ndim == 1')
        tmp = np.zeros((n_nodes, n_nodes)) * np.nan
        for ci in zip(con, *indices):
            tmp[ci[1:]] = ci[0]
        con = tmp
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # remove diagonal (do not show node's self-connectivity)
    np.fill_diagonal(con, np.nan)

    # get the colormap
    if isinstance(colormap, string_types):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, axisbg=facecolor)

    axes.spines['bottom'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)

    tmp = np.empty((n_nodes + 4, n_nodes + 4)) * np.nan
    tmp[2:-2, 2:-2] = con
    con = tmp

    h = axes.imshow(con, cmap=colormap, interpolation='nearest', vmin=vmin,
                    vmax=vmax)

    nodes = np.empty((n_nodes + 4, n_nodes + 4, 4)) * np.nan
    for i in range(n_nodes):
        nodes[i + 2, 0, :] = node_colors[i]
        nodes[i + 2, -1, :] = node_colors[i]
        nodes[0, i + 2, :] = node_colors[i]
        nodes[-1, i + 2, :] = node_colors[i]
    axes.imshow(nodes, interpolation='nearest')

    if colorbar:
        cb = plt.colorbar(h, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    # Draw node labels
    if show_names:
        for i, name in enumerate(node_names):
            axes.text(-1, i + 2, name, size=fontsize_names,
                      rotation=0, rotation_mode='anchor',
                      horizontalalignment='right', verticalalignment='center',
                      color=textcolor)
            axes.text(i + 2, len(node_names) + 4, name, size=fontsize_names,
                      rotation=90, rotation_mode='anchor',
                      horizontalalignment='right', verticalalignment='center',
                      color=textcolor)

    axes.format_coord = partial(_plot_connectivity_matrix_nodename, con=con,
                                node_names=node_names)

    return fig, axes


def plot_connectivity_inoutcircles(con, seed, node_names, facecolor='black',
                                   textcolor='white', colormap='hot',
                                   title=None, fontsize_suptitle=14, fig=None,
                                   subplot=(121, 122), **kwargs):
    """Visualize effective connectivity with two circular graphs, one for
    incoming, and one for outgoing connections.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.loria.fr/~rougier/coding/recipes.html

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    seed : int | str
        Index or name of the seed node. Connections towards and from that node
        are displayed. The seed can be changed by clicking on a node in
        interactive mode.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    colormap : str | (str, str)
        Colormap to use for coloring the connections. Can be a tuple of two
        strings, in which case the first colormap is used for incoming, and the
        second colormap for outgoing connections.
    title : str
        The figure title.
    fontsize_suptitle : int
        Font size to use for title.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : (int, int) | (3-tuple, 3-tuple)
        Location of the two subplots for incoming and outgoing connections.
        E.g. 121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    **kwargs :
        The remaining keyword-arguments will be passed directly to
        plot_connectivity_circle.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes_in : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    axes_out : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt

    n_nodes = len(node_names)

    if any(isinstance(seed, t) for t in string_types):
        try:
            seed = node_names.index(seed)
        except ValueError:
            from difflib import get_close_matches
            close = get_close_matches(seed, node_names)
            raise ValueError('{} is not in the list of node names. Did you '
                             'mean {}?'.format(seed, close))

    if seed < 0 or seed >= n_nodes:
        raise ValueError('seed={} is not in range [0, {}].'
                         .format(seed, n_nodes - 1))

    if type(colormap) not in (tuple, list):
        colormap = (colormap, colormap)

    # Default figure size accomodates two horizontally arranged circles
    if fig is None:
        fig = plt.figure(figsize=(8, 4), facecolor=facecolor)

    index_in = (np.array([seed] * n_nodes),
                np.array([i for i in range(n_nodes)]))
    index_out = index_in[::-1]

    fig, axes_in = plot_connectivity_circle(con[seed, :].ravel(), node_names,
                                            indices=index_in,
                                            colormap=colormap[0], fig=fig,
                                            subplot=subplot[0],
                                            title='incoming', **kwargs)

    fig, axes_out = plot_connectivity_circle(con[:, seed].ravel(), node_names,
                                             indices=index_out,
                                             colormap=colormap[1], fig=fig,
                                             subplot=subplot[1],
                                             title='outgoing', **kwargs)

    if title is not None:
        plt.suptitle(title, color=textcolor, fontsize=fontsize_suptitle)

    return fig, axes_in, axes_out
