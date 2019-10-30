import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from brain_RNN import *


# Plot train and validation loss
def plot_train_val_loss(out_path, title, dpi=800, yscale=None, ylim=None):
    data = pd.read_csv(os.path.join(out_path, 'loss.csv'))
    sns.scatterplot(x='epoch', y='loss', hue='step', data=data, s=15)
    plt.legend(ncol=4)

    if yscale:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    output_fname = os.path.join(out_path, f'training_{title}.png')
    plt.savefig(output_fname, dpi=dpi)
    plt.clf()


# Plot result
def plot_result(y_tensor, result_tensor, minmax_y, out_path, out_fname):
    y_arr = normalize_tensor(y_tensor, minmax_y)
    result_arr = normalize_tensor(result_tensor, minmax_y)
    plt.plot(y_arr, result_arr, '.')
    plt.ylim(minmax_y)
    plt.title(out_fname)
    output_fname = os.path.join(out_path, 'result.png')
    plt.savefig(output_fname)
    plt.clf()

