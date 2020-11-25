import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from loss_functions import KL, calculate_iou
from helper import get_dm

matplotlib.use('pdf')


def plot_histogram_and_image(hist_pred, hist_true, img, tile_name, out_dir=None, volume_weighted=False):
    """
    plotting the predicted histogram on the top of original histogram, next to the image of original tile
    """
    img = img.astype(np.uint8)

    index = np.arange(len(hist_pred)) + 0.5

    KL_div = KL(hist_true, hist_pred)

    iou = calculate_iou(hist_true, hist_pred)

    dm_true = get_dm(hist_true, volume_weighted=volume_weighted)
    dm_pred = get_dm(hist_pred, volume_weighted=volume_weighted)

    # Create Figure and Axes instances
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6.5))

    # add title of the whole figure
    fig.suptitle('Comparison of Distribution\n%s' % (tile_name), fontsize=18)

    ax1.bar(index, hist_true, width=1.0, label='true histogram')
    ax1.bar(index, hist_pred, width=1.0, alpha=0.5, label='predicted histogram')
    ax1.legend(fontsize=14)

    # axis labels
    ax1.set_xlabel('Grain diameter [cm]', fontsize=16)

    if volume_weighted:
        ax1.set_ylabel('Relative volume', fontsize=16)
    else:
        ax1.set_ylabel('Relative frequency', fontsize=16)

    # x ticks labels
    group_labels = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0, 1.2, 1.5, 2.0]) * 100
    group_labels = np.array(group_labels, dtype=np.int)
    ax1.set_xticks(np.arange(len(group_labels)))
    ax1.set_xticklabels(group_labels, rotation='vertical')

    ax1.text(  # position text relative to Axes
        0.98, 0.82, 'KL: %.2f' % (KL_div),
        ha='right', va='top',
        transform=ax1.transAxes,
        fontsize=16
    )

    ax1.text(  # position text relative to Axes
        0.98, 0.76, 'IoU: %.2f' % (iou),
        ha='right', va='top',
        transform=ax1.transAxes,
        fontsize=16
    )

    ax1.text(  # position text relative to Axes
        0.98, 0.70, 'dm true: %.2f cm' % (dm_true),
        ha='right', va='top',
        transform=ax1.transAxes,
        fontsize=16
    )
    ax1.text(  # position text relative to Axes
        0.98, 0.64, 'dm pred: %.2f cm' % (dm_pred),
        ha='right', va='top',
        transform=ax1.transAxes,
        fontsize=16
    )

    ax2.set_xticks(())
    ax2.set_yticks(())

    ax2.imshow(img)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        plt.savefig(os.path.join(out_dir, '{}.png'.format(tile_name)), bbox_inches='tight')
        plt.close(fig)  # close the figure

