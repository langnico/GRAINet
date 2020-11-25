# coding=utf-8
import os
import numpy as np
import argparse
import json
import codecs
from glob import glob
from keras import losses
from skimage.transform import resize

import preprocessing as prepro
from loss_functions import mean_IOU, emd, reverseKL, JSD, mse_weighted, mae_weighted

# set up printing values from numpy arrays to not use exponential representation
np.set_printoptions(suppress=True)


def str2bool(v):
    """
    Helper function to read input arguments as boolean.
    :param v: string
    :return: bool
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_parser():
    """
    Setup an argparse object to parse input arguments.
    Default values are set to run the random cross-validation with 10 folds to regress a scalar output (mean diameter).
    :return: ArgumentParser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", default=1, help='0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.', type=int)

    parser.add_argument("--experiment_type", default='randCV', help="str: type of experiment", choices=['randCV', 'bankCV'])
    parser.add_argument("--data_npz_path", default=None, help="str: path to npz data file")
    parser.add_argument("--bank_names_path", default=None, help="str: path to txt file with bank names need for bankCV")
    parser.add_argument("--randCV_indices_path", default=None, help="str: path to npy file with list of fold indices for random crossval")
    parser.add_argument("--experiment_dir", default=None, help="str: output path to experiment directory")

    # inference
    parser.add_argument("--image_path", default=None, help="str: path full ortho image tif for inference")
    parser.add_argument("--inference_path", default=None, help="str: out path for inference")

    parser.add_argument("--img_rows", default=500, help="int: image rows", type=int)
    parser.add_argument("--img_cols", default=200, help="int: image columns", type=int)
    parser.add_argument("--channels", default=3, help="int: image channels", type=int)
    parser.add_argument("--bins", default=21, help="int: number of histogram bins", type=int)

    parser.add_argument("--batch_size", default=8, help="int: batch size", type=int)
    parser.add_argument("--nb_epoch", default=100, help="int: batch size", type=int)
    parser.add_argument("--base_lr", default=0.0003, help="float: base learning rate", type=float)
    parser.add_argument("--loss_key", default='mse', help="str: name of the loss that is optimized", choices=['kld', 'emd', 'mse', 'mae', 'iou', 'rkl', 'jsd', 'msew', 'maew'])

    parser.add_argument("--test_fold_index", default=0, help='subset (fold) index used for testing', type=int)
    parser.add_argument("--test_bank_name", default=None, help="str: name of test bank")

    parser.add_argument("--volume_weighted", type=str2bool, nargs='?', const=True, default=False, help="Bool: Train/test on volume weighted pdfs.")
    parser.add_argument("--output_dm", type=str2bool, nargs='?', const=True, default=True, help="Bool: Network outputs dm instead of distribution.")
    parser.add_argument("--downsample_factor", default=1., help="float: downsample factor. factor=2 reduces the image width and height by factor 2 (rounding down)", type=float)

    return parser


def get_loss_dict():
    """
    Definition of loss functions (evaluation metrics) used in this project.
    All losses expect two inputs (y_true, y_pred), i.e., ground truth and predictions.
    :return: dictionary with loss functions
    """
    loss_dict = {'kld': losses.kullback_leibler_divergence,
                 'rkl': reverseKL,
                 'jsd': JSD,
                 'iou': mean_IOU,
                 'mse': 'mean_squared_error',
                 'mae': 'mean_absolute_error',
                 'emd': emd,
                 'msew': mse_weighted,
                 'maew': mae_weighted}
    return loss_dict


def transform_histogram(cdf_all):
    """
    Transforms all ground truth given as cumulative distribution function (CDF) to probability density function (PDF).
    :param cdf_all: list of cdf arrays
    :return: np array containing pdf ground truth
    """
    pdf_list = []
    for cdf in cdf_all:
        pdf_list.append(cdf2pdf(cdf))
    pdf_arrays = np.array(pdf_list, dtype=np.float32)
    return pdf_arrays


def cdf2pdf(cdf):
    """
    Transform CDF to PDF.
    :param cdf: array
    :return: pdf with num_bins = len(cdf) - 1
    """
    pdf = np.diff(cdf)
    return np.array(pdf, dtype=np.float32)


def save_history(path, history):
    """
    Save keras training history as json.
    :param path: (string) output file path .json
    :param history: keras history output
    """
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)


def load_history(path):
    """
    Load saved keras history from json file.
    :param path: (string) output file path .json
    :return: history as dictionary
    """
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n


def cum_histo(hist):
    """
    Create a cumulative histogram. (pdf to cdf)
    :param hist: array (histogram, pdf)
    :return: array (cdf)
    """
    hist_cum = np.cumsum(hist)
    return hist_cum


def save_histograms(file_name, predicted_histo, names):
    """
    Save all the predicted histograms in the excel file
    :param file_name: (string) output file path
    :param predicted_histo: array
    :param names: tile names
    """
    # check if histograms have length 22
    predicted_histo = [np.concatenate((np.array([0]), histo)) for histo in predicted_histo if len(histo) == 21]

    names = np.expand_dims(names, axis=1)
    data_to_save = np.hstack((names, predicted_histo))

    labels = "Name,d: 0.00 m,d: 0.01 m,d: 0.02m,d: 0.03 m,d: 0.04 m,d: 0.06 m,d: 0.08 m,d: 0.10 m,d: 0.12 m,d: 0.15 m,d: 0.20 m,d: 0.25 m,d: 0.30 m,d: 0.35 m,d: 0.40 m,d: 0.50 m,d: 0.60 m,d: 0.80 m,d: 1.0 m,d: 1.2 m,d: 1.5 m,d: 2.0 m"

    format_specifier = ['%f'] * 22
    format_specifier.insert(0, '%s')

    np.savetxt(file_name, data_to_save, fmt=format_specifier, delimiter=",", header=labels)


def save_dm(file_name, predicted_dm, names):
    """
    Save all the mean diameters in the excel file.
    :param file_name: (string) output file path
    :param predicted_dm: array
    :param names: tile names
    """
    data_to_save = np.stack((names, predicted_dm), axis=-1)

    labels = "Name,dm [cm]"
    format_specifier = ['%s', '%f']

    np.savetxt(file_name, data_to_save, fmt=format_specifier, delimiter=",", header=labels)


def get_relative_volume_distribution(relative_frequency_hist):
    """
    Transform frequency distribution to volume distribution.
    Each bin frequency is weighted by a volume proxy. Here this is the bin center grain size squared.
    :param relative_frequency_hist: array (pdf)
    :return: array (volume weighted pdf)
    """

    # upper limits of all classes
    edges = np.array(
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80,
         1.0, 1.2, 1.5, 2.0], dtype=float)

    n_bins = len(edges) - 1
    # mean diameter of the class i
    dmi = np.zeros(n_bins)
    for d in range(0, n_bins):
        dmi[d] = (edges[d+1] + edges[d]) / 2.

    # percentage mass of class i, before additional empirical corrections
    mean_bin_volume = np.power(dmi, 2)
    volume_hist = relative_frequency_hist * mean_bin_volume
    relative_volume_hist = volume_hist / np.sum(volume_hist)

    return relative_volume_hist


def relFreq2relVolume(histograms):
    """
    Wrapper to transform all frequency distributions to volume distributions.
    :param histograms: array
    :return: array
    """
    rel_vol_hist_all = []
    for hist in histograms:
        rel_vol_hist_all.append(get_relative_volume_distribution(hist))
    return np.array(rel_vol_hist_all)


# calculate a mean diameters of a histogram
def get_dm(delta_qi, volume_weighted=False):
    """
    Calculate the mean diameters from pdf (either frequency or already volume weighted).
    We compute the mean diameter according to Fehr (1987).

    Fehr, R. (1987). Einfache Bestimmung der Korngrössenverteilung von Geschiebematerial
    mit Hilfe der Linienzahlanalyse. Schweizer Ingenieur und Architekt, 38(87), 1104-1109.

    :param delta_qi: array (pdf frequency or volume)
    :param volume_weighted: bool (if true, the input delta_qi the volume weighting is skipped)
    :return: float (mean diameter in cm)
    """

    if len(delta_qi) == 21:
        # ad a zero bin
        delta_qi = np.concatenate((np.array([0]), delta_qi))

    # delta_qi is a traditional histogram
    delta_qi = np.expand_dims(delta_qi, axis=1)

    # upper limits of all classes
    d_grenz = np.array(
        [0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80,
         1.0, 1.2, 1.5, 2.0], dtype=float)
    anz_intervall = len(d_grenz)

    # mean diameter of the class i
    dmi = np.zeros([anz_intervall, 1])

    for d in range(anz_intervall):
        if d == 0:
            dmi[d] = d_grenz[d] / 2
        else:
            dmi[d] = (d_grenz[d] + d_grenz[d - 1]) / 2

    if not volume_weighted:
        # percentage mass of class i, before additional empirical corrections
        delta_qi_dmi2 = delta_qi * np.power(dmi, 2)
        delta_pi = delta_qi_dmi2 / np.sum(delta_qi_dmi2)
    else:
        # input histogram is already volume weighted
        delta_pi = delta_qi

    # cumulative percentage of mass of class i, before additional empirical corrections
    pi = np.zeros([anz_intervall])

    for d in range(anz_intervall):
        if d == 0:
            pi[d] = 0
        else:
            pi[d] = pi[d - 1] + delta_pi[d]

    # cumulative percentage mass of class i, first corection: admitting 25% of the stones are smaller and not measured
    pi_c = np.zeros([anz_intervall, 1])

    for d in range(anz_intervall):
        if d == 0:
            pi_c[d] = 0
        else:
            pi_c[d] = 0.25 + 0.75 * pi[d]

    # Second correction - calculating an empirical curve, the so called Fuller-Curve
    pFU_1 = np.zeros([anz_intervall - 1, 1])
    diff_p = np.zeros([anz_intervall - 1, 1])

    diff_vg = 100

    # added new line
    rel_ind = 0

    for d in range(anz_intervall - 1):
        if d == 0:
            pass
        else:
            pFU_1[d] = np.sqrt(d_grenz[d + 1] / (d_grenz[d] / np.power(pi_c[d], 2)))
            diff_p[d] = np.abs(pi_c[d] - pFU_1[d])
            if pi_c[d] > 0:
                if pi_c[d] < 0.99999999999999:
                    if diff_p[d] < diff_vg:
                        rel_ind = d
                        diff_vg = diff_p[d]
    if rel_ind < 2:
        rel_ind = 2
    elif rel_ind > 7:
        rel_ind = 7

    piFU = np.zeros([anz_intervall, 1])

    for d in range(anz_intervall):
        piFU[d] = np.sqrt(d_grenz[d] / (d_grenz[rel_ind - 1] / np.power(pi_c[rel_ind - 1], 2)))

    pi_rel = np.zeros([anz_intervall, 1])

    pi_rel[0:rel_ind] = piFU[0:rel_ind]
    pi_rel[rel_ind::] = pi_c[rel_ind::]

    # index_1 = np.where(pi_rel > 0.99)[0][0] + 1
    delta_pi_rel = np.zeros([anz_intervall, 1])
    # zaehler = 0
    for d in range(anz_intervall - 1):
        if d == 0:
            delta_pi_rel[d] = pi_rel[d] - 0
        else:
            delta_pi_rel[d] = pi_rel[d] - pi_rel[d - 1]

    dm_t = np.multiply(delta_pi_rel, dmi)
    dm = np.sum(dm_t)

    # returning median diameter in cm
    return dm * 100


def make_crossval_ramdom_train_val_test_split(data, indices_list_path, test_fold):
    """
    Split data into random subsets using predefined sample indices.
    :param data: dictionary containing arrays
    :param indices_list_path: npy file path with stored fold sample indices
    :param test_fold: test fold index
    :return: dictionary
    """
    # load list of fold indices
    indices_list = list(np.load(indices_list_path, allow_pickle=True))

    test_indices = indices_list[test_fold]

    # all except indices from fold test fold  i
    trainval_indices = indices_list[:test_fold] + indices_list[test_fold+1:]
    #print('len training indices list: ', len(trainval_indices))
    trainval_indices = np.concatenate(trainval_indices)
    if not set(test_indices).isdisjoint(set(trainval_indices)):
        raise ValueError("TEST indices overlap with TRAINING indices.")

    # split training into 80% train and 20% val
    train_indices = trainval_indices[:int(0.9 * len(trainval_indices))]
    val_indices = trainval_indices[int(0.9 * len(trainval_indices)):]

    fold_data = {}

    # transform the histogram from the original cumulative to the traditional one
    histogram_all_transformed = transform_histogram(data['histograms'])

    # images
    fold_data['X_train'] = data['images'][train_indices]
    fold_data['X_val'] = data['images'][val_indices]
    fold_data['X_test'] = data['images'][test_indices]

    # labels
    fold_data['Y_train'] = histogram_all_transformed[train_indices]
    fold_data['Y_val'] = histogram_all_transformed[val_indices]
    fold_data['Y_test'] = histogram_all_transformed[test_indices]

    # names of tiles
    fold_data['N_train'] = data['tile_names'][train_indices]
    fold_data['N_val'] = data['tile_names'][val_indices]
    fold_data['N_test'] = data['tile_names'][test_indices]

    # dm
    fold_data['D_train'] = data['dm'][train_indices]
    fold_data['D_val'] = data['dm'][val_indices]
    fold_data['D_test'] = data['dm'][test_indices]

    fold_data['MEAN_train'], fold_data['STD_train'] = prepro.get_mean_and_std_per_channel(fold_data['X_train'])

    return fold_data


def is_tile_from_bank(tile_name, bank_name):
    """
    Check if image tile comes from particular river bank by name.
    :param tile_name: string
    :param bank_name: string
    :return: bool (True if tile comes from bank)
    """
    return tile_name[:-12] == bank_name


def make_crossval_banks_train_val_test_split(data, test_bank_name):
    """
    Split data into train, val, test, holding out a complete river bank for testing (by name).
    Used for generalization experiments.
    :param data: dictionary
    :param test_bank_name: string (river bank to test on)
    :return: dictionary
    """

    vec_is_tile_from_bank = np.vectorize(is_tile_from_bank)

    test_indices = np.squeeze(np.argwhere(vec_is_tile_from_bank(data['tile_names'], test_bank_name)))
    print('{}: {} test samples'.format(test_bank_name, test_indices.shape))

    if len(test_indices) == 0:
        raise ValueError('no test samples found, check test_bank_name: {}'.format(test_bank_name))

    # all except indices from fold test fold  i
    trainval_indices = np.argwhere(~vec_is_tile_from_bank(data['tile_names'], test_bank_name))
    print('trainval_indices.shape', trainval_indices.shape)
    trainval_indices = np.concatenate(trainval_indices)
    print('trainval_indices.shape', trainval_indices.shape)

    if not set(list(test_indices)).isdisjoint(set(list(trainval_indices))):
        raise ValueError("TEST indices overlap with TRAINING indices.")

    # split training into 80% train and 20% val
    train_indices = trainval_indices[:int(0.9 * len(trainval_indices))]
    val_indices = trainval_indices[int(0.9 * len(trainval_indices)):]

    fold_data = {}

    # transform the histogram from the original cumulative to the traditional one
    histogram_all_transformed = transform_histogram(data['histograms'])

    # images
    fold_data['X_train'] = data['images'][train_indices]
    fold_data['X_val'] = data['images'][val_indices]
    fold_data['X_test'] = data['images'][test_indices]

    # labels
    fold_data['Y_train'] = histogram_all_transformed[train_indices]
    fold_data['Y_val'] = histogram_all_transformed[val_indices]
    fold_data['Y_test'] = histogram_all_transformed[test_indices]

    # names of tiles
    fold_data['N_train'] = data['tile_names'][train_indices]
    fold_data['N_val'] = data['tile_names'][val_indices]
    fold_data['N_test'] = data['tile_names'][test_indices]

    # dm
    fold_data['D_train'] = data['dm'][train_indices]
    fold_data['D_val'] = data['dm'][val_indices]
    fold_data['D_test'] = data['dm'][test_indices]

    fold_data['MEAN_train'], fold_data['STD_train'] = prepro.get_mean_and_std_per_channel(fold_data['X_train'])

    print('train samples: ', len(fold_data['X_train']))
    print('val   samples: ', len(fold_data['X_val']))
    print('test  samples: ', len(fold_data['X_test']))

    return fold_data


def downsample_images(images, factor):
    """
    Reduce image resolution by factor (>1).
    :param images: array
    :param factor: float (e.g. factor=2 doubles the ground sampling distance.)
    :return: array (downsampled images)
    """
    images_out = []
    for i in range(len(images)):
        image = images[i]
        image_resized = resize(image, (image.shape[0] // factor, image.shape[1] // factor), anti_aliasing=True)
        images_out.append(image_resized)
        
    return np.array(images_out)


def create_k_fold_split_indices(data, out_path, num_folds=10):
    np.random.seed(21)
    # shuffle the images randomly to create train, val and test sets
    indices = np.arange(data['images'].shape[0])
    # always generate the same random numbers with random seed for testing the code
    np.random.shuffle(indices)
    # split indices int n folds:
    indices_list = np.array_split(indices, num_folds)
    # indices_list is saved as array of type object, because not all folds have the exact same length.
    np.save(out_path, indices_list)
    return np.array(indices_list, dtype=object)


def wrapper_make_data_split(experiment_type, data_npz_path, test_bank_name, test_fold_index, randCV_indices_path,
                            volume_weighted=False,
                            output_dm=False,
                            downsample_factor=1.):    
    """
    Wrapper to create train, val, test splits.
    Either random cross-validation (randCV) or bank cross-validation (bankCV).
    :param experiment_type: string (choices=['randCV', 'bankCV']
    :param data_npz_path: path to data
    :param test_bank_name: string (bank to test if experiment_type='bankCV')
    :param test_fold_index: int (fold index to test if experiment_type='randCV')
    :param randCV_indices_path: path to saved data split with sample indices
    :param volume_weighted: bool (True, if targets are volume weighted histograms)
    :param output_dm: bool (True, if targets are scalars, i.e. mean diameters)
    :param downsample_factor: float (if input image should be artificially downsampled)
    :return: dictionary
    """
    
    data_all = np.load(data_npz_path, allow_pickle=True)

    if experiment_type == 'randCV':
        data = make_crossval_ramdom_train_val_test_split(data=data_all, indices_list_path=randCV_indices_path, test_fold=test_fold_index)

    elif experiment_type == 'bankCV':
        data = make_crossval_banks_train_val_test_split(data=data_all, test_bank_name=test_bank_name)

    else:
        raise ValueError("experiment type: '{}' is not defined.".format(experiment_type))

    if volume_weighted:
        # labels
        print('converting labels to relative volume distributions (volume pdf)...')
        data['Y_train'] = relFreq2relVolume(data['Y_train'])
        data['Y_val'] = relFreq2relVolume(data['Y_val'])
        data['Y_test'] = relFreq2relVolume(data['Y_test'])

    if output_dm:
        # output dm instead of histogram
        data['Y_train'] = data['D_train']
        data['Y_val'] = data['D_val']
        data['Y_test'] = data['D_test']

    # For testing the effect of different input image resolutions
    if downsample_factor != 1:
        data['X_train'] = downsample_images(data['X_train'], factor=downsample_factor)
        data['X_val'] = downsample_images(data['X_val'], factor=downsample_factor)
        data['X_test'] = downsample_images(data['X_test'], factor=downsample_factor)

    return data


class NumpyEncoder(json.JSONEncoder):
    """
    Convert nparrays to lists for json writing
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def collect_cv_data(parent_dir, loss_keys=('emd', 'iou', 'kld', 'mae', 'mse', 'maew', 'msew', 'rkl', 'jsd')):
    """
    Collect cross-validation outputs from all test folds.
    :param parent_dir: path to directory containing all fold directories.
    :param loss_keys: list of loss/metric keys to evaluate
    :return results_dict: evaluation metrics for all folds
    :return avg_results_dict: mean and std of metrics over all folds
    :return dm_results_dict: true and predicted mean diameters
    """
    loss_keys = sorted(loss_keys)

    results_dict = {}
    out_dir = os.path.join(parent_dir, 'collected')

    dm_results_dict = {}

    for loss_key in loss_keys:
        print('loss: ', loss_key)
        results_dict[loss_key] = {}
        dm_results_dict[loss_key] = {}
        predicted_dm_all = np.array([])
        true_dm_all = np.array([])

        # get all fold directories
        fold_dirs = glob(os.path.join(parent_dir, 'loss_{}'.format(loss_key), 'testfold_*'))
        print('Number of fold directories:', len(fold_dirs))
        # get all results keys and initialize lists
        tmp_dict = json.load(open(fold_dirs[0] + '/test_metrics.json', 'r'))
        for k in tmp_dict.keys():
            results_dict[loss_key][k] = []
        tmp_dict = json.load(open(fold_dirs[0] + '/test_dm_metrics.json', 'r'))
        for k in tmp_dict.keys():
            results_dict[loss_key][k] = []

        for fdir in fold_dirs:
            test_metrics = json.load(open(fdir + '/test_metrics.json', 'r'))
            for k in test_metrics.keys():
                results_dict[loss_key][k].append(test_metrics[k])
            test_dm_metrics = json.load(open(fdir + '/test_dm_metrics.json', 'r'))
            for k in test_dm_metrics.keys():
                results_dict[loss_key][k].append(test_dm_metrics[k])

            # collect dm results
            pred_dm = np.load(os.path.join(fdir, 'dm_pred.npy'))
            true_dm = np.load(os.path.join(fdir, 'dm_true.npy'))

            predicted_dm_all = np.concatenate((predicted_dm_all, pred_dm))
            true_dm_all= np.concatenate((true_dm_all, true_dm))

        dm_results_dict[loss_key]['dm_true'] = true_dm_all
        dm_results_dict[loss_key]['dm_pred'] = predicted_dm_all

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    json.dump(results_dict, open(os.path.join(out_dir, 'results_dict.json'), 'w'))


    # average/ std over folds
    avg_results_dict = {}
    for loss_key in loss_keys:
        avg_results_dict[loss_key] = {}
        for metric in results_dict[loss_key].keys():
            avg_results_dict[loss_key][metric] = {}

    for loss_key in loss_keys:
        for metric in avg_results_dict[loss_key].keys():
            # convert lists to numpy
            results_dict[loss_key][metric] = np.array(results_dict[loss_key][metric], dtype=np.float32)
            # reduce arrays to mean and std
            avg_results_dict[loss_key][metric]['mean'] = float(np.mean(results_dict[loss_key][metric]))
            avg_results_dict[loss_key][metric]['std'] = float(np.std(results_dict[loss_key][metric]))
    json.dump(avg_results_dict, open(os.path.join(out_dir, 'avg_results_dict.json'), 'w'))
    json.dump(dm_results_dict, open(os.path.join(out_dir, 'dm_results_dict.json'), 'w'), cls=NumpyEncoder)

    return results_dict, avg_results_dict, dm_results_dict


# per table row: create a list of strings
def dict_to_table(avg_results_dict, headers, losses=None):
    table = []
    if losses is None: 
        losses = headers
        
    for loss in losses:
        row_mean_list = []
        row_std_list = []
        row_mean_list.append(loss.replace('_', ' '))
        row_std_list.append('')
        for metric in headers:
            row_mean_list.append("${:.2f}$".format(avg_results_dict[loss][metric]['mean']))
            row_std_list.append("$({:.2f})$".format(avg_results_dict[loss][metric]['std']))
        table.append(row_mean_list)
        table.append(row_std_list)
    return table


