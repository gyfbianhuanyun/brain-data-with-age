import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Plot train and validation loss
def plot_train_val_loss(output_path, title, dpi=800, yscale=None, ylim=None):
    data = pd.read_csv('{}/loss_{}.csv'.format(output_path, title))
    sns.scatterplot(x='epoch', y='loss', hue='k_fold', style='step', legend='full',
                    palette='Set2', data=data, s=15)
    plt.legend(ncol=4)
    #plt.plot(
        #idx_arr, train_loss_arr, idx_arr, valid_loss_arr, linewidth=linewidth)

    if yscale:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    output_fname = f'{output_path}/training_{title}.png'
    plt.savefig(output_fname, dpi=dpi)
    plt.clf()


# Plot result
def plot_result(y_tensor, result_tensor, minmax_y, output_path, out_fname):
    y_arr = normalize_tensor(y_tensor, minmax_y)
    result_arr = normalize_tensor(result_tensor, minmax_y)
    plt.plot(y_arr, result_arr, '.')
    plt.ylim(minmax_y)
    plt.title(out_fname)
    output_fname = f'{output_path}/result_{out_fname}.png'
    plt.savefig(output_fname)
    plt.clf()

