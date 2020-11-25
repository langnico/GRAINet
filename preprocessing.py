import numpy as np

# set up printing values from numpy arrays to not use exponential representation
np.set_printoptions(suppress=True)


def get_mean_and_std_per_channel(img_list):
    """
    compute mean and std per image channel.
    :param img_list: list of arrays
    :return: mean and std
    """
    # mean of all images per channel
    mean = [img_list[..., i].mean() for i in range(img_list.shape[-1])]
    mean = [[[mean[0], mean[1], mean[2]]]]
    # std of all images per channel
    std = [img_list[..., i].std() for i in range(img_list.shape[-1])]
    std = [[[std[0], std[1], std[2]]]]
    return mean, std


def normalize_img_per_channel(image, mean_train, std_train):
    """
    normalize image per channel with to z-scores (mean=0, std=1)
    :param image: array
    :param mean_train: channel means
    :param std_train: channel stds
    :return: normalized image
    """
    img_rows = image.shape[0]
    img_cols = image.shape[1]
    img_zero_mean = image - np.tile(mean_train, [img_rows, img_cols, 1])
    img_norm = np.divide(img_zero_mean, np.tile(std_train, [img_rows, img_cols, 1]))
    return img_norm


def normalize_images_per_channel(images, mean_train, std_train, out_dtype='float32'):
    """
    Normalize all images per channel.
    :param images: array
    :param mean_train: channel means
    :param std_train: channel stds
    :param out_dtype: string (default='float32')
    :return: array (normalized images)
    """
    images_norm = []
    for i in range(images.shape[0]):
        img = images[i, :, :, :]
        assert len(img.shape) == 3
        img_norm = normalize_img_per_channel(img, mean_train, std_train)
        images_norm.append(img_norm)
    return np.asarray(images_norm, out_dtype)


