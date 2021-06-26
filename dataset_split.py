# This code was inspired by: https://arxiv.org/pdf/2010.06099.pdf
# Code retrieved from https://github.com/felipefariax/sbss/blob/master/sbss.py
import numpy as np
from scipy.spatial import distance
import copy


def dataset_splitter(x, y, n_splits, n_classes, distance_name='euclidean', encode_output=False):
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
    :return: the list containing the data subsets (numpy arrays) and a list of the labels associated with each instance
    of the subsets. The distribution of the labels is the same for all the subsets.
    """
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
    print(y_counts)
    min_groups = np.min(y_counts)

    if np.all(n_splits > y_counts):
        raise ValueError("n_splits=%d cannot be greater than the number of members in each class." % n_splits)
    if n_splits > min_groups:
        print(("The least populated class in y has only %d members, which is less than n_splits=%d."
               % (min_groups, n_splits)))

    # array with k rows and n_samples cols. Each column belongs to the same class
    folds_list = [[] for _ in range(n_splits)]
    fold_col_lb = []

    for k in range(n_classes):
        print("Working on class: ", k)
        idx_k = y.squeeze() == k
        new_x = copy.deepcopy(copied_x[idx_k, :])

        while len(new_x) > 0:
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


# if __name__ == '__main__':
#    example()
