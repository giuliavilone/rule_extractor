import numpy as np
from scipy.spatial import distance
import copy


def number_split_finder(y, max_row, min_row=10000):
    """
    Determine the number of subsets (or splits) depending on the number of instances per each input class, the maximum
    and minimum numbers of instances per class.
    :param y: list/array of output labels
    :param max_row: maximum number of instance per subset
    :param min_row: minimum number of instance per subset. This is set to avoid an infinite loop when the numbers of
                    samples per class do not have a denominator in common (they are not divisible for the same number)
    :return: a dictionary with the number of subsets (or splits) per class
    """
    unique, counts = np.unique(y, return_counts=True)
    data_length = dict(zip(unique, counts))
    ret = {x: 1 for x in data_length.keys()}
    for key, value in data_length.items():
        # Starting from the lowest number of splits that will divide the dataset into n subsets smaller than max_row
        split = max(int(np.ceil(value / max_row)), 2)
        find_split = True
        if value <= max_row:
            find_split = False
        while find_split:
            if np.mod(value, split) == 0 and np.round(len(y) / split) < max_row:
                ret[key] = split
                find_split = False
            else:
                # Terminating the loop if the number of splits is too big to generate too small subsets
                if np.round(len(y) / split) <= min_row:
                    ret[key] = split
                    find_split = False
                else:
                    split += 1
    return ret


def data_filler(in_array, n_splits):
    """
    Filling the in_array with extra samples to reach n_splits rows
    :param in_array: input array of samples
    :param n_splits: number of total rows that in_array must have
    :return:in_array
    """
    new_rows = np.random.randint(0, len(in_array), n_splits - len(in_array))
    for i in new_rows:
        instance = np.array([in_array[i, :]])
        in_array = np.append(in_array, instance, axis=0)
    return in_array


def dataset_splitter(x, y, n_splits, n_classes, distance_name='euclidean', encode_output=False, data_filling='discard'):
    """
    The input dataset x is split into n subset so that the most similar instances are assigned to different subsets. In
    this way, each subset represents the entire distribution of the initial dataset.
    :param x: input dataset (pandas dataframe) with the independent variables
    :param y: input list containing the labels of the input dataset
    :param n_splits: n of subsets to be generated
    :param n_classes: number of unique labels
    :param distance_name: the distance metric to be used to calculate the similarity between samples. The viable options
    are: 'braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’,
    ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    :param encode_output: the input labels must be encoded into integer numbers if they are strings.
    :param data_filling: how to fill missing instances in the last iteration of the loop in case the number of
    instances is not a multiple of the number of splits. Two options are available: 1) 'discard' means that the last
    data must be discarded, and 2) 'repeat' in case the extra samples must be a repetition of the remaining samples.
    :return: the list containing the data subsets (numpy arrays) and a list of the labels associated with each instance
    of the subsets. The distribution of the labels is the same for all the subsets.
    """
    # This code was inspired by: https://arxiv.org/pdf/2010.06099.pdf
    # Code retrieved from https://github.com/felipefariax/sbss/blob/master/sbss.py
    copied_x = copy.deepcopy(x)
    copied_x = copied_x.to_numpy()
    n_splits_range = [i for i in range(n_splits-1)]
    if encode_output:
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to map the classes so that they are encoded
        # by order of appearance: 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        n_classes = len(y_idx)
    else:
        y_encoded = y

    y_counts = np.bincount(y_encoded)
    print('y_counts: ', y_counts)
    min_groups = np.min(y_counts)

    if np.all(n_splits > y_counts):
        raise ValueError("n_splits=%d cannot be greater than the number of members in each class." % n_splits)
    if n_splits > min_groups:
        print(("The least populated class in y has only %d members, which is less than n_splits=%d."
               % (min_groups, n_splits)))
    if data_filling not in ['discard', 'repeat']:
        raise ValueError("data_filling=%d is not a valid option ('discard', 'repeat')." % data_filling)

    # array with k rows and n_samples cols. Each column belongs to the same class
    folds_list = [[] for _ in range(n_splits)]
    fold_col_lb = []

    for k in range(n_classes):
        print("Working on class: ", k)
        idx_k = y_encoded.squeeze() == k
        new_x = copy.deepcopy(copied_x[idx_k, :])

        while len(new_x) > 0:
            if len(new_x) < len(n_splits_range):
                if data_filling == 'discard':
                    break
                else:
                    new_x = data_filler(new_x, n_splits)

            instance = np.array([new_x[0, :]])
            new_x = new_x[1:, :]
            sum_distances = distance.cdist(new_x, instance, metric=distance_name)

            # Take the pivot sample, which has the shortest distance to the sample under consideration
            nearby_samples_ids = np.argpartition(sum_distances, tuple(n_splits_range), axis=0)[n_splits_range]
            nearby_samples_ids = list(nearby_samples_ids.reshape(-1))
            nearby_samples = new_x[nearby_samples_ids, :]
            new_x = np.delete(new_x, nearby_samples_ids, axis=0)
            nearby_samples = np.append(nearby_samples, instance, axis=0)

            fold_col_lb.append(k)

            # Shuffle samples to ensure stochastic assignment to each split
            np.random.shuffle(nearby_samples)

            for split_idx in np.arange(n_splits):
                folds_list[split_idx].append(nearby_samples[split_idx])

    folds = np.array(folds_list)
    folds_labels = np.array(fold_col_lb)

    return folds, folds_labels


def example():
    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target

    data = np.array(
        [
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [11, 11, 11, 1],
            [0, 11, 11, 1],
            [0, 0, 11, 1],
            [111, 111, 111, 1],
            [0, 111, 111, 1],
            [0, 0, 111, 1],
            [2, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 0, 2, 0],
            [22, 22, 22, 0],
            [0, 22, 22, 0],
            [0, 0, 22, 0],
            [222, 222, 222, 0],
            [0, 222, 222, 0],
            [0, 0, 222, 0],
        ]
    )

    x = data[:, :-1]
    y = data[:, -1]
    splits, labels = dataset_splitter(x=x, y=y, n_splits=3, n_classes=2)
    print(splits)
    print(labels)


def array_append(a1, a2, axis=0):
    if a1 is None:
        a1 = a2
    else:
        a1 = np.append(a1, a2, axis=axis)
    return a1


def cluster_finder(x, key, instances, split_instance, split_labels, max_row, distance_name):
    """
    :param x:
    :param key:
    :param instances:
    :param split_instance:
    :param split_labels:
    :param max_row:
    :param distance_name:
    :return:
    """
    if len(split_instance) < max_row and len(x) > 0:
        nearby_samples = None
        for instance in instances:
            if len(x) > 4:
                sum_distances = distance.cdist(x, np.array([instance]), metric=distance_name)
                # Take the pivot sample, which has the shortest distance to the sample under consideration
                nearby_samples_ids = np.argpartition(sum_distances, (0, 1), axis=0)[[0, 1]]
                nearby_samples_ids = list(nearby_samples_ids.reshape(-1))
                nearby_samples = array_append(nearby_samples, x[nearby_samples_ids, :])
                x = np.delete(x, nearby_samples_ids, axis=0)
            else:
                nearby_samples = array_append(nearby_samples, x)
                x = np.array([])
                break
        split_instance = np.append(split_instance, nearby_samples, axis=0)
        split_labels += [key] * len(nearby_samples)
        split_instance, split_labels, x = cluster_finder(x, key, nearby_samples, split_instance, split_labels,
                                                         max_row, distance_name)
    return split_instance, split_labels, x


def dataset_splitter_new(x, y, n_splits, distance_name='euclidean'):
    """
    The input dataset x is split into n subset so that the most similar instances are assigned to the same subset. In
    this way, the cluster analysis will be facilitated. Each class is split separately.
    :param x: input dataset (pandas dataframe) with the independent variables
    :param y: input list containing the labels of the input dataset
    :param n_splits: n of subsets to be generated
    :param distance_name: the distance metric to be used to calculate the similarity between samples. The viable options
    are: 'braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’,
    ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    :return: the list containing the data subsets (numpy arrays) and a list of the labels associated with each instance
    of the subsets. The distribution of the labels is the same for all the subsets.
    """
    copied_x = copy.deepcopy(x)
    copied_x = copied_x.to_numpy()

    # array with k rows and n_samples cols. Each column belongs to the same class
    folds_list = []
    fold_col_lb = []

    for key, value in n_splits.items():
        idx_k = y.squeeze() == key
        new_x = copy.deepcopy(copied_x[idx_k, :])
        if value > 1:
            max_row = int(np.ceil(len(new_x) / (value + 1)))
            while len(new_x) > 0:
                instances = np.array([new_x[0, :]])
                new_x = new_x[1:, :]
                split_list, split_lbs, new_x = cluster_finder(new_x, key, instances, instances, [key], max_row,
                                                              distance_name)
                folds_list.append(split_list)
                fold_col_lb.append(split_lbs)
        else:
            folds_list.append(new_x)
            fold_col_lb.append([key] * len(new_x))

    folds = np.array(folds_list)
    folds_labels = np.array(fold_col_lb)

    return folds, folds_labels


# if __name__ == '__main__':
#    example()
