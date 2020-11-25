from osgeo import gdal
import numpy as np

import os
from keras.layers import Input

import preprocessing as prepro
from resnet_architecture import FCN_grainsize
import helper


def read_rgb(ds, rgb_indices=(1, 2, 3)):
    """
    Read RGB bands and returns a 3D numpy array with shape: [height, width, channels]
    :param ds: gdal raster dataset
    :param rgb_indices: gdal band index starts at 1 (not 0)
    :return: numpy rgb bands
    """
    bands_all = []
    for i in rgb_indices:
        band = ds.GetRasterBand(i)
        band_array = band.ReadAsArray()
        print('band_array.shape: ', band_array.shape)
        bands_all.append(band_array)
    bands_all = np.array(bands_all, dtype=np.float32)
    bands_all = np.moveaxis(bands_all, source=0, destination=2) # channels last
    return bands_all


def get_tiles(ds, tile_rows=500, tile_cols=200):

    img = read_rgb(ds=ds)
    img_rows, img_cols = img.shape[0:2]

    print('img_rows, img_cols:', img_rows, img_cols)
    print('tile_rows: {}, tile_cols: {}'.format(tile_rows, tile_cols))
    
    rows_range = np.arange(0, int(img_rows / tile_rows) * tile_rows, tile_rows)
    cols_range = np.arange(0, int(img_cols / tile_cols) * tile_cols, tile_cols)

    # round to integers for selecting number of pixels
    rows_range = np.array(np.round(rows_range), dtype=np.int32)
    cols_range = np.array(np.round(cols_range), dtype=np.int32)

    print('num tiles row: ', len(rows_range))
    print('num tiles col: ', len(cols_range))
    print('total tiles :', len(rows_range) * len(cols_range))

    tiles = []
    for i in rows_range:
        for j in cols_range:
            tile = img[i:i + int(tile_rows), j:j + int(tile_cols), :]
            if tile.shape[:2] == (int(tile_rows), int(tile_cols)):
                tiles.append(tile)
    tiles = np.array(tiles)
    print('tiles.shape:', tiles.shape)
    pred_shape = (len(rows_range), len(cols_range))
    print('pred_shape: ', pred_shape)
    return tiles, pred_shape


def save_array_as_geotif(out_path, ref_ds, array, x_res, y_res, out_width, out_height):
    out_bands = 1
    geotransform = list(ref_ds.GetGeoTransform())   # ([your_top_left_x, 30, 0, your_top_left_y, 0, -30])
    # set output resolution
    geotransform[1] = x_res  # east to west
    geotransform[-1] = -y_res  # north to south

    dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, out_width, out_height, out_bands, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(ref_ds.GetProjection())
    dst_ds.GetRasterBand(1).WriteArray(array)  # write r-band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def run_prediction_orthophoto(args):
    GSD_orig = 0.0025
    print('downsample_factor: ', args.downsample_factor)

    if not os.path.exists(args.inference_path):
        os.makedirs(args.inference_path)

    ds = gdal.Open(args.image_path)

    # adjust original tile size (GSD 0.0025) for downsampling factor
    args.img_rows /= args.downsample_factor
    args.img_cols /= args.downsample_factor

    tiles, pred_shape = get_tiles(ds=ds, tile_rows=args.img_rows, tile_cols=args.img_cols)

    # load preprocessing statistics
    train_MEAN = np.load(os.path.join(args.experiment_dir, 'train_MEAN.npy'))
    train_STD = np.load(os.path.join(args.experiment_dir, 'train_STD.npy'))

    X_test_prepro = prepro.normalize_images_per_channel(images=tiles, mean_train=train_MEAN, std_train=train_STD,
                                                        out_dtype='float32')

    # initialize the model with input of proper shape
    input_shape = (int(args.img_rows), int(args.img_cols), args.channels)  # for tensorflow: channels last
    img_input = Input(shape=input_shape)
    # load model
    model = FCN_grainsize(img_input=img_input, bins=args.bins, output_scalar=args.output_dm)

    # load trained weights
    weights_filepath_val = os.path.join(args.experiment_dir, 'weights_best_val.h5')
    model.load_weights(weights_filepath_val)

    # predict
    predictions = model.predict(X_test_prepro)
    print('predictions.shape: ', predictions.shape)

    if not args.output_dm:
        # get dms
        dm_preds = []
        for pred in predictions:
            dm_preds.append(helper.get_dm(pred))
        dm_preds = np.array(dm_preds)
    else:
        # copy predictions (dm output)
        dm_preds = np.array(predictions)

    print('dm_preds.shape:', dm_preds.shape)

    # reshape predictions
    dm_pred_reshaped = np.reshape(dm_preds, newshape=pred_shape)
    print('dm_pred_reshaped.shape: ', dm_pred_reshaped.shape)

    save_array_as_geotif(out_path=os.path.join(args.inference_path, 'dm_pred.tif'), ref_ds=ds, array=dm_pred_reshaped,
                         x_res=GSD_orig * args.downsample_factor * args.img_cols,
                         y_res=GSD_orig * args.downsample_factor * args.img_rows,
                         out_width=pred_shape[1],
                         out_height=pred_shape[0])

    np.save(os.path.join(args.inference_path, 'predictions.npy'), predictions)
    np.save(os.path.join(args.inference_path, 'dm_pred_2D.npy'), dm_pred_reshaped)

    return predictions, dm_pred_reshaped, read_rgb(ds=ds)


if __name__ == "__main__":

    # set parameters
    parser = helper.setup_parser()
    args, unknown = parser.parse_known_args()

    run_prediction_orthophoto(args=args)

