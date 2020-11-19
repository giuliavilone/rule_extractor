# From https://github.com/nakumgaurav/XAI-TREPAN-for-Regression/blob/master/descision_tree.py
import numpy as np
from scipy.stats import gaussian_kde, entropy
import copy


class Oracle:
    def __init__(self, network, num_classes, dataset, discrete_feature):
        self.network = network
        self.num_classes = num_classes
        self.X = dataset
        self.y = self.get_oracle_labels(self.X)
        self.disc = discrete_feature
        self.num_features = self.X.shape[-1]
        self.construct_training_distribution()
        self.train_dist = []

    def construct_training_distribution(self):
        """Get the density estimates for each feature using a kernel density estimator.
        Any estimator could be used as described here:
        https://ned.ipac.caltech.edu/level5/March02/Silverman/paper.pdf """
        for i in range(self.num_features):
            data = self.X[:, i]
            try:
                kernel = gaussian_kde(data, bw_method='silverman')
            except:
                # If the categorical variables have all the same code (like [0,0,0]) the gaussian_kde goes into
                # error and it is necessary to add a small noise
                data = data + 0.001 * np.random.randn(data.shape[0])
                kernel = gaussian_kde(data, bw_method='silverman')
            self.train_dist.append(kernel)

    def generate_instance(self, constraint):
        """Given the constraints that an instance must satisfy, generate an instance """
        instance = np.zeros(self.num_features)
        for feature_no in range(self.num_features):
            while True:
                # According to the paper, if the variable is categorical the value must be selected according to
                # the frequency of the discrete values
                if feature_no in self.disc:
                    values, prob = np.unique(self.X[:, feature_no], return_counts=True)
                    prob = prob / sum(prob)
                    sampled_val = np.random.choice(values, 1, p=prob)[0]
                else:
                    sampled_val = self.train_dist[feature_no].resample(1)[0]
                if constraint.satisfy_feature(sampled_val, feature_no):
                    break

            instance[feature_no] = sampled_val

        return instance

    def generate_instances(self, constraint, size):
        new_instances = []
        for i in range(size):
            new_instances.append(self.generate_instance(constraint))

        return np.array(new_instances)

    def get_oracle_labels(self, samples):
        """Returns the label predicated by the oracle network for example"""
        onehot = self.network.predict(samples)
        return np.argmax(onehot, axis=1)


class Constraint:
    def __init__(self):
        self.constraint = []

    def add_rule(self, rule):
        self.constraint.append(rule)

    def satisfy_feature(self, value, feat_no):
        """ Given a feature value, check whether feat_no feature satisfies the constraint """
        ans = True
        for rule in self.constraint:
            feature_no, symbol, thresh = rule.split(" ")
            feature_no = int(feature_no[1:])
            if feature_no != feat_no:
                continue

            thresh = float(thresh)
            if symbol == "<=":
                ans &= value <= thresh
            elif symbol == ">":
                ans &= value > thresh

        return ans

    def satisfy(self, instance):
        """Given an instance, check whether it satisfies the constraint """
        ans = True
        for feat_no in range(len(instance)):
            ans &= self.satisfy_feature(instance[feat_no], feat_no)

        return ans

    def get_constrained_features(self):
        """ Gives the list of indices for features on which constraint rules are present """
        feature_indices = []
        for rule in self.constraint:
            feature_no, _, _ = rule.split(" ")
            feature_no = int(feature_no[1:])
            feature_indices.append(feature_no)
        return list(set(feature_indices))


class Node:
    def __init__(self, data, labels, constraints):
        self.data = data
        self.labels = labels
        self.constraints = constraints
        self.num_features = data.shape[1]
        self.left = None
        self.right = None
        self.dominant = self.get_dominant_class(labels)
        self.misclassified = self.get_misclassified_count(labels)
        # This is to rule out features that have been already used to generate splits to reach this node
        self.blacklisted_features = self.constraints.get_constrained_features()

    @staticmethod
    def get_dominant_class(labels):
        """
        This function returns the "dominant" class of this node, i.e., the class with the highest count of the examples
        in this node.
        """
        class_counts = {}
        # get count for all labels
        for label in labels:
            if label not in class_counts:
                # insert in counter
                class_counts[label] = 0
            class_counts[label] += 1

        # get the class with the max count
        max_class = max(class_counts, key=class_counts.get)
        return max_class

    def get_misclassified_count(self, labels):
        """
        Get the number of training examples misclassified by this node, if it were a leaf.
        This is nothing but the number of examples not belonging to the dominant class.
        This value is used to calculate the "priority" of a node, which determines when to explore/split a node.
        """
        num_misclassified = 0
        for label in labels:
            if label != self.dominant:
                num_misclassified += 1
        return num_misclassified


class Tree:
    def __init__(self, oracle):
        # Create root node with constrains set equal to [], initial data equal to the entire training dataset
        # and labels predicted by the oracle model
        self.root = self.construct_node(self.initial_data, self.initial_labels, Constraint())
        self.oracle = oracle
        self.initial_data = oracle.X
        self.initial_labels = oracle.y
        self.num_examples = len(oracle.X)
        self.tree_params = {"tree_size": 50, "split_min": 100, "num_feature_splits": 15}
        self.num_nodes = 0
        self.max_levels = 0

    @staticmethod
    def construct_node(data, labels, constraints):
        """ Input Args - data: the training data that this node has
            Output Args - A Node variable
        """
        return Node(data, labels, constraints)

    @staticmethod
    def is_leaf(node) -> object:
        return node.left is None and node.right is None

    @staticmethod
    def get_entropy(labels):
        """
        Takes a list of labels, and calculates the entropy.
        """
        if len(labels) == 0:
            return 0
        labels_prob = [labels.count(i) / len(labels) for i in set(labels)]
        return entropy(labels_prob, base=2)

    def get_gain(self, labels, split_1, split_2):
        """
        Calculates the entropy gain from the two splits
        """
        orig_ent = self.get_entropy(labels)
        after_ent = (self.get_entropy(labels[split_1]) * (sum(split_1) / len(labels)) +
                     self.get_entropy(labels[split_2]) * (sum(split_2) / len(labels)))
        return orig_ent - after_ent

    def binary_info_gain(self, feature, threshold, samples, labels):
        """
        Takes a feature and a threshold, examples and their
        labels, and find the best feature and breakpoint to split on to maximise
        information gain.
        Assumes only two classes. Would need to be altered if more are required.
        """
        # Get two halves of threshold
        split1 = samples[:, feature] >= threshold
        split2 = np.invert(split1)
        # Get entropy after split (remembering to weight by no of examples in each
        # half of split)
        return self.get_gain(labels, split1, split2)

    def mofn_info_gain(self, mofntest, samples, labels):
        """
        Takes an m-of-n test with a set of samples and labels, and calculates the information gain provided by the test.

        Structure of an m-of-n test object: (m, [(feat_1, thresh_1, greater_1)...(feat_n, thresh_n, greater_n)])
        Where m = number of tests that must be passed (i.e. m in m-of-n)
        feat_i = the feature of the ith test
        thresh_i = the threshold of the ith test
        greater_i = a boolean: If true, value must be >= than threshold to pass the test.
                               If false, it must be < threshold.
        """
        # Unpack the tests structure
        m = mofntest[0]
        sep_tests = mofntest[1]
        # List comprehension to generate a boolean index that tells us which samples
        # passed the test.
        split_test = np.array([samples[:, sep[0]] >= sep[1] if sep[2] else
                              samples[:, sep[0]] < sep[1] for sep in sep_tests])
        # Now check whether the number of tests passed per sample is higher than m
        split1 = sum(split_test) >= m
        split2 = np.invert(split1)
        # Calculate and return gain
        return self.get_gain(labels, split1, split2)

    @staticmethod
    def expand_mofn_test(test, feature, threshold, greater, increment_m):
        """
        Constructs and returns a new m-of-n test using the passed test and
        other parameters.
        """
        # Check for feature redundancy
        for feat in test[1]:
            if feature == feat[0]:
                if greater == feat[2]:
                    # Just return the unmodified existing test if we'd add a threshold with same feature and sign
                    return test
                else:
                    # Also just return the same if the two tests would overlap
                    if (greater and threshold <= feat[1]) or (not greater and threshold >= feat[1]):
                        return test
        # If we didn't find redundancy, actually create the test
        if increment_m:
            new_m = test[0] + 1
        else:
            new_m = test[0]
        new_feats = list(test[1])
        new_feats.append((feature, threshold, greater))
        return [new_m, new_feats]

    def get_priority(self, node):
        reach_n = float(len(node.data)) / self.num_examples
        print(f"reach_n={reach_n}")
        fidelity_n = self.get_fidelity(node)
        print(f"fidelity_n={fidelity_n}")
        # Multiplied by -1 to order the nodes with the highest priority in decreasing order
        priority = -1 * reach_n * (1 - fidelity_n)
        return float(priority)

    def get_fidelity(self, node):
        l2e = 1 - (float(node.misclassified) / self.num_examples)
        return l2e

    def build_tree(self):
        """Main method which builds the tree and returns the root
        through which the entire tree can be accessed"""
        import queue as Q
        node_queue = Q.PriorityQueue(maxsize=self.tree_params["tree_size"])
        # node_queue = Q.PriorityQueue()

        node_queue.put((self.get_priority(self.root), self.num_nodes, self.root), block=False)
        self.num_nodes += 1

        while not node_queue.empty() and self.num_nodes <= self.tree_params["tree_size"]:
            print("num_nodes = ", self.num_nodes)
            priority, _, node = node_queue.get()
            node = self.add_instances(node)
            node = self.split(node)
            if node.left is None and node.right is None:  # meaning that the node is a leaf
                continue
            else:
                left_prio = self.get_priority(node.left)
                right_prio = self.get_priority(node.right)
                print("left_prio=", left_prio)
                print("right_prio=", right_prio)

                node_queue.put((left_prio, self.num_nodes + 1, node.left), block=False)
                node_queue.put((right_prio, self.num_nodes + 2, node.right), block=False)
                self.num_nodes += 2

        return self.root

    def add_instances(self, node):
        """Query the oracle to add more instances to the node if required """
        num_instances = len(node.data)
        s_min = self.tree_params["split_min"]
        if num_instances >= s_min:
            return node

        num_new_instances = s_min - num_instances
        new_instances = self.oracle.generate_instances(node.constraints, size=num_new_instances)

        new_data = np.zeros(shape=(s_min, self.oracle.num_features))
        new_data[:num_instances] = node.data
        new_data[num_instances:] = new_instances
        node.data = new_data
        node.labels = self.oracle.get_oracle_labels(new_data)

        return node

    def get_best_split(self, node):
        """
        Return the feature with the best split among those that have not been already used to reach the node
        :param node: node under analysis which contains also the list of the features previously split to reach it
        :return: the feature with the best split and its best split point
        """
        min_mse = float("inf")
        best_split_point = None
        best_feat = None

        for i in range(self.oracle.num_features):
            if i in node.blacklisted_features:
                continue

            split_point, mse = self.feature_split(node.data[:, i])

            if mse < min_mse:
                best_feat = i
                best_split_point = split_point
                min_mse = mse

        return best_feat, best_split_point

    def split(self, node):
        """Decide the best split and split the node. In case it is not possible to determine the best split, the
         node is set as a leaf"""
        best_feat, best_split_point = self.get_best_split(node)
        if best_feat is None:
            node.left = None
            node.right = None
        else:
            left_ind = node.data[:, best_feat] <= best_split_point
            right_ind = node.data[:, best_feat] > best_split_point

            left_constraints = copy.deepcopy(node.constraints)
            right_constraints = copy.deepcopy(node.constraints)
            left_rule = f"x{best_feat} <= {best_split_point}"
            right_rule = f"x{best_feat} > {best_split_point}"

            left_constraints.add_rule(left_rule)
            right_constraints.add_rule(right_rule)

            node.left = self.construct_node(node.data[left_ind], node.labels[left_ind], left_constraints)
            node.right = self.construct_node(node.data[right_ind], node.labels[right_ind], right_constraints)
            node.split_rule = left_rule

        return node

    def calc_mse(self, data):
        """
        Calculate the minimum squared error
        :param data:
        :return: minimum squared error value
        """
        mean = np.mean(data)
        return np.mean((data - mean) ** 2)

    def feature_split(self, feature_data):
        """
        Find the best binary split for the input feature
        :param feature_data: training data related to a single independent feature
        :return: the point where the data must be split and its minimum squared error
        """
        split_points = np.linspace(start=min(feature_data), stop=max(feature_data),
                                   num=self.tree_params["num_feature_splits"]
                                   )[1:-1]
        min_mse = float("inf")
        mse = float("inf")
        best_split_point = None

        for split_point in split_points:
            data_left = feature_data[feature_data <= split_point]
            data_right = feature_data[feature_data > split_point]
            mse = self.calc_mse(data_left) + self.calc_mse(data_right)

            if mse < min_mse:
                best_split_point = split_point
                min_mse = mse

        return best_split_point, mse

    def predict(self, instance, root):
        if self.is_leaf(root):
            return root.dominant
        if root.constraints.satisfy(instance):
            return self.predict(instance, root.left)
        else:
            return self.predict(instance, root.right)

    def assign_levels(self, root, level):
        root.level = level
        if level > self.max_levels:
            self.max_levels = level

        if root.left is not None:
            self.assign_levels(root.left, level + 1)
        if root.right is not None:
            self.assign_levels(root.right, level + 1)

    def print_tree_levels(self, root):
        for level in range(self.max_levels + 1):
            self.print_tree(root, level)

    def print_tree(self, root, level):
        if self.is_leaf(root):
            if level == root.level:
                print('node level:', level)
                print(root.dominant)
        else:
            if level == root.level:
                print(root.split_rule)

            if root.left is not None:
                self.print_tree(root.left, level)
            if root.right is not None:
                self.print_tree(root.right, level)

    def leaf_values(self, root, ret_list=None):
        if ret_list is None:
            ret_list = []
        if self.is_leaf(root):
            ret_list += [root.level]
        if root.left is not None:
            self.leaf_values(root.left, ret_list)
        if root.right is not None:
            self.leaf_values(root.right, ret_list)
        return ret_list
