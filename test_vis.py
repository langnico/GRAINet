import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import json

import helper
import plots


def create_plots(args):
    # load data
    data = helper.wrapper_make_data_split(experiment_type=args.experiment_type,
                                          data_npz_path=args.data_npz_path,
                                          test_bank_name=args.test_bank_name,
                                          test_fold_index=args.test_fold_index,
                                          randCV_indices_path=args.randCV_indices_path,
                                          volume_weighted=args.volume_weighted,
                                          output_dm=args.output_dm,
                                          downsample_factor=args.downsample_factor)

    # load predictions
    predictions = np.load(os.path.join(args.experiment_dir, 'predictions.npy'))

    # save all the predicted images to figures and excel file
    original_labels_all = data['Y_test'].copy()
    original_images_all = data['X_test'].copy()
    original_names_all = data['N_test'].copy()
    original_dm_all = data['D_test'].copy()

    # all the differences between dms
    diff_all = []

    # all the predicted dms
    predicted_dm = []
    true_dm = []

    # loop through all test images
    for j in range(predictions.shape[0]):
        dm_true = original_dm_all[j]

        if args.output_dm:
            dm_pred = predictions[j]
        else:
            if not args.volume_weighted:
                dm_pred = helper.get_dm(predictions[j])
                dm_true_cal = helper.get_dm(original_labels_all[j])
            else:
                dm_pred = helper.get_dm(predictions[j], volume_weighted=args.volume_weighted)
                dm_true_cal = helper.get_dm(original_labels_all[j], volume_weighted=args.volume_weighted)

            # check
            diff_true_source = dm_true - dm_true_cal
            if diff_true_source > 0.01:
                print('ground truth dm is not identical something is wrong with the computation')

        diff = dm_pred - dm_true

        predicted_dm.append(dm_pred)
        true_dm.append(dm_true)
        diff_all.append(diff)

        out_str = 'diff: %.2f ' % (diff) + 'predicted dm: %.2f ' % (dm_pred) + 'original dm: %.2f ' % (
            dm_true) + 'tile name: ' + original_names_all[j]

        with open(os.path.join(args.experiment_dir, 'test_output_per_sample.txt'), 'a') as file:
            file.write("\n")
            file.write(out_str)

    diff_all = np.array(diff_all, np.float32)
    predicted_dm = np.array(predicted_dm, np.float32).squeeze()
    true_dm = np.array(true_dm, np.float32)

    diff_all_rel = diff_all / true_dm

    # absolute error
    dm_mae = np.mean(np.abs(diff_all))
    dm_mse = np.mean(np.square(diff_all))
    dm_me = np.mean(diff_all)

    # relative dm error
    dm_mae_rel = np.mean(np.abs(diff_all_rel))
    dm_mse_rel = np.mean(np.square(diff_all_rel))
    dm_me_rel = np.mean(diff_all_rel)

    test_dm_metrics = {'dm_mae': float(dm_mae),
                       'dm_mse': float(dm_mse),
                       'dm_me': float(dm_me),
                       'dm_mae_rel': float(dm_mae_rel),
                       'dm_mse_rel': float(dm_mse_rel),
                       'dm_me_rel': float(dm_me_rel)}

    json.dump(test_dm_metrics, open(os.path.join(args.experiment_dir, 'test_dm_metrics.json'), "w"))

    if not args.output_dm:
        print('saving figures...')
        # save figures
        for l in range(predictions.shape[0]):
            plots.plot_histogram_and_image(predictions[l], original_labels_all[l], original_images_all[l],
                                           original_names_all[l],
                                           out_dir=os.path.join(args.experiment_dir, 'test_plots'),
                                           volume_weighted=args.volume_weighted)

        print('figures saved')
        # save histograms
        helper.save_histograms(file_name=os.path.join(args.experiment_dir, 'predicted_histograms.csv'),
                               predicted_histo=predictions, names=original_names_all)

        print('saving tabels...')
        # save cumulative histograms
        predictions_transformed = []

        for m in range(predictions.shape[0]):
            hist_new = helper.cum_histo(predictions[m])
            predictions_transformed.append(hist_new)

        helper.save_histograms(file_name=os.path.join(args.experiment_dir, 'predicted_cumulative_histograms.csv'),
                               predicted_histo=predictions_transformed, names=original_names_all)

    # save dms
    print('predicted_dm.shape: ', predicted_dm.shape)
    print('original_names_all.shape: ', original_names_all.shape)
    helper.save_dm(file_name=os.path.join(args.experiment_dir, 'predicted_dms.csv'), predicted_dm=predicted_dm,
                   names=original_names_all)
    print('tables saved')

    # save true and predicted dm as npy
    np.save(file=os.path.join(args.experiment_dir, 'dm_pred.npy'), arr=predicted_dm)
    np.save(file=os.path.join(args.experiment_dir, 'dm_true.npy'), arr=true_dm)

    # plot dm true vs pred
    mi, ma = 0, max(np.max(true_dm), np.max(predicted_dm))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_dm, predicted_dm)
    plt.xlabel('Ground truth dm [cm]')
    plt.ylabel('Predicted dm [cm]')
    plt.xlim((mi, ma))
    plt.ylim((mi, ma))
    plt.axis('equal')
    plt.grid()
    # perfect calibration line
    plt.plot([mi, ma], [mi, ma], "k-", zorder=0)
    plt.savefig(os.path.join(args.experiment_dir, 'dm_scatter.png'), bbox_inches='tight', dpi=300)


if __name__ == "__main__":

    parser = helper.setup_parser()
    args, unknown = parser.parse_known_args()

    create_plots(args=args)

