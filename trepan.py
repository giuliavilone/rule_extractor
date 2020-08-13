# From https://github.com/nakumgaurav/XAI-TREPAN-for-Regression/blob/master/descision_tree.py
import numpy as np
from scipy.stats import gaussian_kde
import copy


class Oracle:
    def __init__(self, network, num_classes, dataset):
        self.network = network
        self.num_classes = num_classes
        self.X = dataset
        self.y = self.get_oracle_labels(self.X)
        self.num_features = self.X.shape[-1]
        self.construct_training_distribution()

    def construct_training_distribution(self):
        """Get the density estimates for each feature using a kernel density estimator.
        Any estimator could be used as described here:
        https://ned.ipac.caltech.edu/level5/March02/Silverman/paper.pdf """
        self.train_dist = []

        for i in range(self.num_features):
            data = self.X[:, i]
            kernel = gaussian_kde(data, bw_method='silverman')
            self.train_dist.append(kernel)

    def generate_instance(self, constraint):
        """Given the constraints that an instance must satisfy, generate an instance """
        instance = np.zeros(self.num_features)
        for feature_no in range(self.num_features):
            while True:
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


class Node():
    def __init__(self, data, labels, constraints):
        self.data = data
        self.labels = labels
        self.constraints = constraints
        self.num_features = data.shape[1]
        self.left = None
        self.right = None
        self.dominant = self.get_dominant_class(labels)
        self.misclassified = self.get_misclassified_count(labels)

        self.blacklisted_features = self.constraints.get_constrained_features()

    def get_dominant_class(self, labels):
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
        max_count = 0
        max_class = 0
        for label in class_counts:
            if class_counts[label] > max_count:
                max_class = label
                max_count = class_counts[label]
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
        self.oracle = oracle
        self.initial_data = oracle.X
        self.initial_labels = oracle.y
        self.num_examples = len(oracle.X)
        self.tree_params = {"tree_size": 25, "split_min": 200, "l1e_threshold": 0.01, "num_feature_splits": 10}
        self.num_nodes = 0
        self.max_levels = 0

    def get_priority(self, node):
        reach_n = float(len(node.data)) / self.num_examples
        print(f"reach_n={reach_n}")
        fidelity_n = self.get_fidelity(node)
        print(f"fidelity_n={fidelity_n}")
        priority = -1 * reach_n * (1 - fidelity_n)
        return priority

    def get_fidelity(self, node):
        # num_corr_predictions = 0
        l2e = 1 - (float(node.misclassified)/self.num_examples)
        # l2e = np.exp(-(predictions - self.oracle.get_oracle_labels(node.data).squeeze()) ** 2)
        #     num_corr_predictions += 1 if l1e <= self.tree_params["l1e_threshold"] else 0
        return l2e

    def build_tree(self):
        """Main method which builds the tree and returns the root
        through which the entire tree can be accessed"""
        import queue as Q
        # node_queue = Q.PriorityQueue(maxsize=self.tree_params["tree_size"])
        node_queue = Q.PriorityQueue()

        self.root = self.construct_node(self.initial_data, self.initial_labels, Constraint())
        self.num_nodes += 1
        node_queue.put((self.get_priority(self.root), self.root), block=False)

        while not node_queue.empty() and self.num_nodes <= self.tree_params["tree_size"]:
            print("num_nodes = ", self.num_nodes)
            priority, node = node_queue.get()
            node = self.add_instances(node)
            node = self.split(node)

            left_prio = self.get_priority(node.left)
            right_prio = self.get_priority(node.right)
            print("left_prio=", left_prio)
            print("right_prio=", right_prio)

            node_queue.put((left_prio, node.left), block=False)
            node_queue.put((right_prio, node.right), block=False)
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
        """Decide the best split and split the node """
        best_feat, best_split_point = self.get_best_split(node)

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
        mean = np.mean(data)
        return np.mean((data - mean) ** 2)

    def feature_split(self, feature_data):
        split_points = np.linspace(start=0, stop=1, num=self.tree_params["num_feature_splits"])[1:-1]
        min_mse = float("inf")
        best_split_point = None

        for split_point in split_points:
            data_left = feature_data[feature_data <= split_point]
            data_right = feature_data[feature_data > split_point]
            mse = self.calc_mse(data_left) + self.calc_mse(data_right)

            if mse < min_mse:
                best_split_point = split_point
                min_mse = mse

        return best_split_point, mse

    def is_leaf(self, node):
        return node.left is None and node.right is None

    def predict(self, instance, root):
        if self.is_leaf(root):
            return root.dominant
        if root.constraints.satisfy(instance):
            return self.predict(instance, root.left)
        else:
            return self.predict(instance, root.right)

    def construct_node(self, data, labels, constraints):
        """ Input Args - data: the training data that this node has
            Output Args - A Node variable
        """
        return Node(data, labels, constraints)

    # print tree

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
                print(root.dominant, " ")
        else:
            if level == root.level:
                print(root.split_rule)

            if root.left is not None:
                self.print_tree(root.left, level)
            if root.right is not None:
                self.print_tree(root.right, level)
