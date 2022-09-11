from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import filter_metrics_means, filter_metrics_sdterr

def add_line(ax, xpos, ypos):
    line = plt.Line2D([ypos, ypos+ .2], [xpos, xpos], color='black', transform=ax.transAxes)
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df):
    xpos = -.2
    scale = 1./df.index.size
    for level in range(df.index.nlevels):
        pos = df.index.size
        for label, rpos in label_len(df.index,level):
            add_line(ax, pos*scale, xpos)
            pos -= rpos
            lypos = (pos + .5 * rpos)*scale
            ax.text(xpos+.1, lypos, label, ha='center', transform=ax.transAxes)
        add_line(ax, pos*scale , xpos)
        xpos -= .2


def performance_heatmap_plot(scores, DIR=None, cmap='Reds', figsize = (15,10)):
    plt.figure(dpi=1200)

    df = pd.DataFrame.from_dict(filter_metrics_means(scores))
    df_stderr = pd.DataFrame.from_dict(filter_metrics_sdterr(scores))

    df = df.T
    df_stderr = df_stderr.T
    annot = np.asarray([["{:.4f}".format(df.iloc[i,j]) +u"\u00B1"+"{:.4f}".format(df_stderr.iloc[i,j]) for j in range(df.shape[1])] for i in range(df.shape[0])])

    norm_df = (df - df.min(0)) / (df.max(0) - df.min(0))

    g = sns.clustermap(norm_df, row_cluster=False, col_cluster=False , cmap=cmap,annot=annot, yticklabels=False, xticklabels=True, linewidths=0.004
                    , linecolor='black', figsize=figsize, fmt='')

    g.ax_heatmap.yaxis.set_ticks_position("left")

    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=10)
    g.ax_heatmap.set_yticks([])
    label_group_bar_table(g.ax_heatmap, df)
    g.cax.set_visible(False)
    if DIR is not None:
        plt.savefig(DIR, bbox_inches = 'tight')
    plt.show()

