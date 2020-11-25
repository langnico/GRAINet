import numpy as np
import os
import json
from time import time

from keras.layers import Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

import preprocessing as prepro
from resnet_architecture import FCN_grainsize, initialize_weights
import helper


def run_train(args):
    """
    A wrapper function to run the training of the CNN.
    :param args: arguments parsed with argparse. See helper.setup_parser() for description.
    """
    loss_dict = helper.get_loss_dict()

    batch_size_val = 8  # number of val/test samples is 96

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    log_dir = os.path.join(args.experiment_dir, "logs", str(time()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load and preprocess data
    data = helper.wrapper_make_data_split(experiment_type=args.experiment_type,
                                          data_npz_path=args.data_npz_path,
                                          test_bank_name=args.test_bank_name,
                                          test_fold_index=args.test_fold_index,
                                          randCV_indices_path=args.randCV_indices_path,
                                          volume_weighted=args.volume_weighted,
                                          output_dm=args.output_dm,
                                          downsample_factor=args.downsample_factor)

    if args.verbose:
        print('train samples: ', len(data['X_train']))
        print('val   samples: ', len(data['X_val']))
        print('test  samples: ', len(data['X_test']))

    # save preprocessing statistics
    np.save(os.path.join(args.experiment_dir, 'train_MEAN.npy'), arr=np.array(data['MEAN_train'], dtype=np.float32))
    np.save(os.path.join(args.experiment_dir, 'train_STD.npy'), arr=np.array(data['STD_train'], dtype=np.float32))

    X_train_prepro = prepro.normalize_images_per_channel(images=data['X_train'], mean_train=data['MEAN_train'],
                                                         std_train=data['STD_train'], out_dtype='float32')
    X_val_prepro = prepro.normalize_images_per_channel(images=data['X_val'], mean_train=data['MEAN_train'],
                                                       std_train=data['STD_train'], out_dtype='float32')

    MEAN_X_train_prepro, STD_X_train_prepro = prepro.get_mean_and_std_per_channel(X_train_prepro)
    MEAN_X_val_prepro, STD_X_val_prepro = prepro.get_mean_and_std_per_channel(X_val_prepro)
    # mean should be approx 0 and std approx 1
    if args.verbose:
        print('TRAIN data prepro:')
        print('mean :{}, std :{} '.format(MEAN_X_train_prepro, STD_X_train_prepro))
        print('VAL data prepro')
        print('mean :{}, std :{} '.format(MEAN_X_val_prepro, STD_X_val_prepro))

    # initialize the model with input of proper shape (for tensorflow: channels last)
    input_shape = (int(args.img_rows / args.downsample_factor), int(args.img_cols / args.downsample_factor),
                   args.channels)
    img_input = Input(shape=input_shape)
    # load model
    model = FCN_grainsize(img_input=img_input, bins=args.bins, output_scalar=args.output_dm)
    if args.verbose:
        # print the summary of the model
        model.summary()

    # calculate the number of batches per epoch
    batches_per_epoch = X_train_prepro.shape[0] // args.batch_size
    validation_steps = X_val_prepro.shape[0] // batch_size_val
    if args.verbose:
        print('number of images per batch: {}'.format(args.batch_size))
        print('batches per train epoch: {}'.format(batches_per_epoch))
        print('validation steps: {}'.format(validation_steps))

    # Create a data generator with data augmentation (horizontal and vertical flipping)
    image_gen_train = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    # image gen without augmentation for validation/test
    image_gen_val = ImageDataGenerator()

    # optimizer
    opt = Adam(lr=args.base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

    # specify the training configuration (optimizer, loss, metrics to monitor)
    if not args.output_dm:
        model.compile(optimizer=opt, loss=loss_dict[args.loss_key],
                      metrics=[loss_dict[l] for l in sorted(loss_dict.keys())])
    else:
        model.compile(optimizer=opt, loss=loss_dict[args.loss_key],
                      metrics=['mean_absolute_error', 'mean_squared_error'])

    # initialize model weights
    initialize_weights(model)

    # save the lowest validation loss
    weights_filepath_val = os.path.join(args.experiment_dir, 'weights_best_val.h5')
    checkpoint_val_loss = ModelCheckpoint(filepath=weights_filepath_val,
                                          monitor='val_loss',
                                          verbose=args.verbose,
                                          save_best_only=True)

    # save the loss and monitor metrics to tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    callbacks_list = [checkpoint_val_loss, tensorboard]

    # train the model with  with data augmentation
    history = model.fit_generator(
        image_gen_train.flow(X_train_prepro, data['Y_train'], batch_size=args.batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=args.nb_epoch,
        callbacks=callbacks_list,
        validation_data=image_gen_val.flow(X_val_prepro, data['Y_val'], batch_size=batch_size_val),
        validation_steps=validation_steps,
        verbose=args.verbose)

    # save the history
    helper.save_history(path=os.path.join(args.experiment_dir, 'history.json'), history=history)

    # save entire model
    model.save(os.path.join(args.experiment_dir, 'model.h5'))


def run_test(args):
    """
    A wrapper function to evaluate the CNN on the test data.
    :param args: arguments parsed with argparse. See helper.setup_parser() for description.
    """
    loss_dict = helper.get_loss_dict()

    # load and preprocess data
    data = helper.wrapper_make_data_split(experiment_type=args.experiment_type,
                                          data_npz_path=args.data_npz_path,
                                          test_bank_name=args.test_bank_name,
                                          test_fold_index=args.test_fold_index,
                                          randCV_indices_path=args.randCV_indices_path,
                                          volume_weighted=args.volume_weighted,
                                          output_dm=args.output_dm,
                                          downsample_factor=args.downsample_factor)

    X_test_prepro = prepro.normalize_images_per_channel(images=data['X_test'], mean_train=data['MEAN_train'],
                                                        std_train=data['STD_train'], out_dtype='float32')

    # initialize the model with input of proper shape
    input_shape = (int(args.img_rows / args.downsample_factor), int(args.img_cols / args.downsample_factor),
                   args.channels)  # for tensorflow: channels last
    img_input = Input(shape=input_shape)
    # load model
    model = FCN_grainsize(img_input=img_input, bins=args.bins, output_scalar=args.output_dm)
    if args.verbose:
        # print the summary of the model
        model.summary()

    # optimizer
    opt = Adam(lr=args.base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
    # specify the training configuration (optimizer, loss, metrics to monitor)
    if not args.output_dm:
        model.compile(optimizer=opt, loss=loss_dict[args.loss_key],
                      metrics=[loss_dict[l] for l in sorted(loss_dict.keys())])
    else:
        model.compile(optimizer=opt, loss=loss_dict[args.loss_key],
                      metrics=['mean_absolute_error', 'mean_squared_error'])

    # load trained weights
    weights_filepath_val = os.path.join(args.experiment_dir, 'weights_best_val.h5')
    model.load_weights(weights_filepath_val)

    # predict
    predictions = model.predict(X_test_prepro)

    # evaluate the model on the test dataset
    print('model evaluation')
    evaluation = model.evaluate(X_test_prepro, data['Y_test'], verbose=args.verbose)

    if not args.output_dm:
        test_metrics = {}
        metric_names = sorted(loss_dict.keys())
        test_out = 'loss: %.4f' % evaluation[0]
        for i in range(len(metric_names)):
            value = evaluation[i + 1]
            # intersection over union is negative for the optimization (minimization)
            if metric_names[i] == 'iou':
                value = float(abs(value))
            test_out += ', %s: %.4f' % (metric_names[i], value)
            test_metrics[metric_names[i]] = value

    else:
        test_out = 'Summary: Loss over the test dataset: %.4f, MAE: %.4f, MSE: %.4f, RMSE: %.4f' % (
            evaluation[0], evaluation[1], evaluation[2], np.sqrt(evaluation[2]))

        # skip evaluation[0]: This is the loss
        test_metrics = {'mae': evaluation[1],
                        'mse': evaluation[2],
                        'rmse': np.sqrt(evaluation[2])}

    print(test_out)
    with open(os.path.join(args.experiment_dir, 'output.txt'), "w") as f:
        f.write(test_out)

    predictions = predictions.squeeze()

    json.dump(test_metrics, open(os.path.join(args.experiment_dir, 'test_metrics.json'), "w"))
    np.save(os.path.join(args.experiment_dir, 'predictions.npy'), arr=predictions)

    return predictions.squeeze(), data['Y_test'].squeeze()

