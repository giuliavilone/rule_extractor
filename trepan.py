# From https://github.com/nakumgaurav/XAI-TREPAN-for-Regression/blob/master/descision_tree.py
# and from https://github.com/KesterJ/TREPAN/blob/master/TREPAN-skeleton.py
import numpy as np
from scipy.stats import gaussian_kde, entropy, mode
from statsmodels.stats.proportion import proportion_confint
import copy
import sys


class Oracle:
    def __init__(self, network, num_classes, dataset, discrete_feature):
        self.network = network
        self.num_classes = num_classes
        self.X = dataset
        self.y = self.get_oracle_labels(self.X)
        self.disc = discrete_feature
        self.num_features = self.X.shape[-1]

    @staticmethod
    def random_choice_sampler(data, size):
        values, prob = np.unique(data, return_counts=True)
        prob = prob / sum(prob)
        return np.random.choice(values, size, p=prob)

    def construct_training_distribution(self, data, feature_no, size):
        """Get the density estimates for each feature using a kernel density estimator.
        Any estimator could be used as described here:
        https://ned.ipac.caltech.edu/level5/March02/Silverman/paper.pdf """
        if feature_no in self.disc:
            sampled_val = self.random_choice_sampler(data, size)
        else:
            try:
                sampled_val = gaussian_kde(data, bw_method='silverman').resample(size)[0]
            except:
                sampled_val = self.random_choice_sampler(data, size)
        return sampled_val

    def generate_instances(self, constraint, size):
        """Given the constraints that an instance must satisfy, generate an instance """
        instance = np.zeros((size, self.num_features))
        unlimited_feat = [i for i in range(self.num_features) if i not in constraint.get_constrained_features()]
        for feature_no in unlimited_feat:
            # According to the paper, if the variable is categorical the value must be selected according to
            # the frequency of the discrete values
            instance[:, feature_no] = self.construct_training_distribution(self.X[:, feature_no], feature_no, size)

        for rule in constraint.constraint:
            rule_passed = rule['passed']
            for counter in range(len(rule['n'])):
                feat_no, thresh, greater = rule['n'][counter]
                greater = np.invert(greater) if not rule_passed else greater
                if greater:
                    data = self.X[:, feat_no][self.X[:, feat_no] >= thresh]
                else:
                    data = self.X[:, feat_no][self.X[:, feat_no] < thresh]
                if len(data) > 0:
                    instance[:, feat_no] = self.construct_training_distribution(data, feat_no, size)
                else:
                    instance[:, feat_no] = self.construct_training_distribution(self.X[:, feat_no], feat_no, size)
        return instance

    def get_oracle_labels(self, samples):
        """Returns the label predicated by the oracle network for example"""
        one_hot = self.network.predict(samples)
        return np.argmax(one_hot, axis=1)


class Constraint:
    def __init__(self):
        self.constraint = []

    def add_rule(self, rule):
        self.constraint.append(rule)

    def satisfy(self, instance):
        """Given an instance, check whether it satisfies the constraint """
        # Initializing at true in case the constrain list is empty
        ans = True
        for rule in self.constraint:
            test_passed = False
            m = rule['m']
            n = len(rule['n'])
            features_passed = 0
            counter = 0
            while (not test_passed) and counter < n:
                feat_no, thresh, greater = rule['n'][counter]
                if (greater and instance[feat_no] >= thresh) or ((not greater) and instance[feat_no] < thresh):
                    features_passed += 1
                if features_passed >= m:
                    test_passed = True
                counter += 1
            ans = test_passed == rule['passed']
        return ans

    def get_constrained_features(self):
        """ Gives the list of indices for features on which constraint rules are present """
        feature_indices = []
        for rule in self.constraint:
            feature_no = [v[0] for v in rule['n']]
            feature_indices += feature_no
        return list(set(feature_indices))


class Node:
    def __init__(self, data, labels, constraints, parent, name):
        self.data = data
        self.labels = labels
        self.constraints = constraints
        self.num_features = data.shape[1]
        self.left = None
        self.right = None
        self.parent = parent
        self.name = name
        self.dominant = self.get_dominant_class(labels)
        self.misclassified = self.get_misclassified_count(labels)
        # This is to rule out features that have been already used to generate splits to reach this node
        self.blacklisted_features = self.constraints.get_constrained_features()

    @staticmethod
    def get_dominant_class(labels):
        """
        This function returns the "dominant" class of this node which corresponds to the mode,
        the class with the highest count of the examples in this node.
        """
        return mode(labels).mode[0]

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
        self.oracle = oracle
        self.initial_data = oracle.X
        self.initial_labels = oracle.y
        self.num_examples = len(oracle.X)
        # Improvement is the percentage by which gain should improve on addition of a new test. (Can be from 1.0+)
        self.tree_params = {"tree_size": 10, "split_min": 1000, "num_feature_splits": 30, 'delta': 0.05, 'eps': 0.05}
        self.conf_interval_low, _ = proportion_confint(self.tree_params['split_min'] * (1 - self.tree_params['eps']),
                                                       self.tree_params['split_min'],
                                                       alpha=self.tree_params['delta'])
        self.num_nodes = 0
        self.max_levels = 0
        self.root = self.construct_node(self.initial_data, self.initial_labels, Constraint(), '', 'root')

    @staticmethod
    def construct_node(data, labels, constraints, parent, name):
        """ Input Args - data: the training data that this node has
            Output Args - A Node variable
        """
        return Node(data, labels, constraints, parent, name)

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
        labels_list = list(labels)
        labels_prob = [labels_list.count(i) / len(labels) for i in set(labels)]
        return entropy(labels_prob, base=2)

    def get_gain(self, labels, split_1, split_2):
        """
        Calculates the entropy gain from the two splits
        """
        orig_ent = self.get_entropy(labels)
        after_ent = (self.get_entropy(labels[split_1]) * (sum(split_1) / len(labels)) +
                     self.get_entropy(labels[split_2]) * (sum(split_2) / len(labels)))
        return orig_ent - after_ent

    def binary_info_gain(self, threshold, samples, labels):
        """
        Takes a feature and a threshold, examples and their
        labels, and find the best feature and breakpoint to split on to maximise
        information gain.
        Assumes only two classes. Would need to be altered if more are required.
        """
        # Get two halves of threshold
        split1 = samples >= threshold
        split2 = np.invert(split1)
        # Get entropy after split (remembering to weight by no of examples in each half of split)
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
        # List comprehension to generate a boolean index that tells us which samples passed the test.
        split_test = np.empty([len(sep_tests), len(samples)])
        for ix in range(len(sep_tests)):
            sep = sep_tests[ix]
            if sep in self.oracle.disc:
                split_test[ix, :] = np.array([samples[:, sep[0]] == sep[1] if sep[2] else samples[:, sep[0]] != sep[1]])
            else:
                split_test[ix, :] = np.array([samples[:, sep[0]] >= sep[1] if sep[2] else samples[:, sep[0]] < sep[1]])

        # Now check whether the number of tests passed per sample is higher than m
        split1 = sum(split_test) >= m
        split2 = np.invert(split1)
        # Calculate and return gain
        return self.get_gain(labels, split1, split2), split1, split2

    def expand_mofn_test(self, test, feature, threshold, samples, labels, original_gain):
        """
        Constructs and returns a new m-of-n test using the passed test and
        other parameters.
        """
        test_changed = 0
        out_test = copy.deepcopy(test)
        new_test = [i for i in test[1] if i[0] != feature]
        candidate_test = [(feature, threshold, True), (feature, threshold, False)]
        new_test_list = [[test[0], new_test + [candidate_test[0]]], [test[0], new_test + [candidate_test[1]]],
                         [test[0] + 1, new_test + [candidate_test[0]]], [test[0] + 1, new_test + [candidate_test[1]]],
                         ]
        gain_list = []
        for item in new_test_list:
            gain, _, _ = self.mofn_info_gain(item, samples, labels)
            gain_list.append(gain)

        if max(gain_list) > original_gain:
            out_test = new_test_list[gain_list.index(max(gain_list))]
            test_changed = 1
            original_gain = max(gain_list)

        return out_test, test_changed, original_gain

    def make_mofn_tests(self, best_test, feat_split_points, samples, labels):
        """
        Finds the best m-of-n test, using a beam width of 2.

        NOTES:
        -NEEDS TO KNOW HOW TO COLLAPSE TESTS WHEN TWO REDUNDANT THINGS ARE PRESENT
            e.g. 2-of {y, z, x, ¬x} -> 1-of {y, z}
        -NEEDS TO KNOW WHICH TESTS WERE ALREADY USED ON THIS BRANCH, AND NOT USE THOSE FEATURES AGAIN - CAN BE DONE
        OUTSIDE FUNCTION BY PASSING SUBSET OF SAMPLES
        -NEEDS TO AVOID USING TWO TESTS ON THE SAME LITERAL e.g. x > 0.5 and x > 0.7
        """
        # Initialise beam with best test and its negation
        init_gain = self.binary_info_gain(best_test[1], samples[:, best_test[0]], labels)
        beam = [[1, [(best_test[0], best_test[1], False)]], [1, [(best_test[0], best_test[1], True)]]]
        current_gains = [init_gain, init_gain]
        beam_changed = 1
        n = 1
        # Set up loop to repeat until beam isn't changed
        while beam_changed:
            print('Test of size %d...' % n)
            n = n + 1
            beam_changed = 0
            # Loop over the current best m-of-n tests in beam
            for ix in range(len(beam)):
                # Loop over the single-features in candidate tests dict
                for feature in feat_split_points:
                    # Loop over the thresholds for the feature
                    for threshold in feat_split_points[feature]:
                        # Add selected feature+threshold to to current test
                        beam[ix], beam_changed, current_gains[ix] = self.expand_mofn_test(beam[ix], feature, threshold,
                                                                                          samples, labels,
                                                                                          current_gains[ix])
            # Set new tests in beam and associated gains
        # Return the best test in beam
        return beam[np.argmax(current_gains)]

    def make_candidate_tests(self, feat_values, labels, feature_number):
        """
        A function that should take one feature and all samples, and return the the possible breakpoints for the
        input feature. These are the midpoints between any two samples that do not have the same label.
        If the feature is discrete, the breakpoints correspond to its unique value.
        """
        # Get unique values for feature
        values = np.unique(feat_values)
        breakpoints = []
        # Loop over values and check if diff classes between values
        if feature_number not in self.oracle.disc:
            for value in range(len(values) - 1):
                # Check if different classes in associated labels, find midpoint if so
                labels1 = labels[feat_values == values[value]]
                labels2 = labels[feat_values == values[value + 1]]
                l1unique = list(np.unique(labels1))
                l2unique = list(np.unique(labels2))
                if l1unique != l2unique or len(l1unique) > 1 or len(l2unique) > 1:
                    midpoint = (values[value] + values[value + 1]) / 2
                    breakpoints.append(midpoint)
            if len(breakpoints) > self.tree_params["num_feature_splits"]:
                idx = np.rint(np.linspace(0, len(breakpoints) - 1, num=self.tree_params["num_feature_splits"])).astype(int)
                breakpoints = [breakpoints[i] for i in idx]
        else:
            breakpoints = values
        # Add list of breakpoints to feature dict
        return breakpoints

    def get_priority(self, node):
        reach_n = len([instance for instance in self.initial_data if node.constraints.satisfy(instance)]) / \
                  self.num_examples
        print(f"reach_n={reach_n}")
        fidelity_n = self.get_fidelity(node)
        print(f"fidelity_n={fidelity_n}")
        # Multiplied by -1 to order the nodes with the highest priority in decreasing order
        priority = -1 * reach_n * (1 - fidelity_n)
        return float(priority)

    def get_fidelity(self, node):
        l2e = 1 - (float(node.get_misclassified_count(self.initial_labels)) / self.num_examples)
        return l2e

    def build_tree(self):
        """Main method which builds the tree and returns the root
        through which the entire tree can be accessed"""
        import queue as Q
        node_queue = Q.PriorityQueue(maxsize=self.tree_params["tree_size"])

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
        best_split_point = None
        best_feat = None
        best_gain = 0
        split_points_per_feat = {}
        # Check if the probability of the most common class is above the lowest limit of the proportion confidence
        # interval determined by the parameters delta and eps. The authors want to check if the node covers with
        # high probability only instances of a single class
        _, prob = np.unique(node.labels, return_counts=True)
        prob = prob / sum(prob)
        if max(prob) > self.conf_interval_low:
            return None

        for i in range(self.oracle.num_features):
            if i in node.blacklisted_features:
                continue

            split_point, test_gain, all_split_points = self.feature_split(node.data[:, i], node.labels, i)
            split_points_per_feat[i] = all_split_points
            if best_gain < test_gain:
                best_feat = i
                best_split_point = split_point
                best_gain = test_gain

        if best_split_point is not None:
            m_of_n_test = self.make_mofn_tests([best_feat, best_split_point],
                                               split_points_per_feat, node.data, node.labels
                                               )
        else:
            m_of_n_test = None
        return m_of_n_test

    def split(self, node):
        """Decide the best split and split the node. In case it is not possible to determine the best split, the
         node is set as a leaf"""
        m_of_n_test = self.get_best_split(node)
        if m_of_n_test is None:
            node.left = None
            node.right = None
        else:
            _, left_ind, right_ind = self.mofn_info_gain(m_of_n_test, node.data, node.labels)

            left_constraints = copy.deepcopy(node.constraints)
            right_constraints = copy.deepcopy(node.constraints)
            left_rule = {'m': m_of_n_test[0], 'n': m_of_n_test[1], 'passed': True}
            right_rule = {'m': m_of_n_test[0], 'n': m_of_n_test[1], 'passed': False}

            left_constraints.add_rule(left_rule)
            right_constraints.add_rule(right_rule)

            node.left = self.construct_node(node.data[left_ind], node.labels[left_ind], left_constraints, node.name,
                                            node.name + '_left')
            node.right = self.construct_node(node.data[right_ind], node.labels[right_ind], right_constraints, node.name,
                                             node.name + '_right')
            node.split_rule = {'m': m_of_n_test[0], 'n': m_of_n_test[1], 'passed': True}

        return node

    def feature_split(self, feature_data, labels, feature_no):
        """
        Find the best binary split for the input feature
        :param labels: array of the output classes related to each input instance
        :param feature_data: training data related to a single independent feature
        :param feature_no: number of the feature under analysis
        :return: the point where the data must be split and its minimum squared error
        """

        split_points = self.make_candidate_tests(feature_data, labels, feature_no)
        best_gain = 0
        best_split_point = None

        for split_point in split_points:
            test_gain = self.binary_info_gain(split_point, feature_data, labels)

            if best_gain < test_gain:
                best_split_point = split_point
                best_gain = test_gain

        return best_split_point, best_gain, split_points

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

    def print_tree_rule(self, root):
        print('-------------------------------------------------------------------------------------------')
        print('Node parent:', root.parent)
        print('Node name:', root.name)
        print('Node dominant: ', root.dominant)
        print('Is node a leaf? ', self.is_leaf(root))
        for constraint in root.constraints.constraint:
            print('Node rule: ', constraint)
        if self.is_leaf(root):
            print('Node level: ', root.level)
            # print('Node split rule: ', root.split_rule)
        else:
            if root.left is not None:
                self.print_tree_rule(root.left)
            if root.right is not None:
                self.print_tree_rule(root.right)

    def leaf_values(self, root, ret_list=None):
        """
        This function return the the number of nodes between each leaf and the root. This should correspond
        to the number of antecedents of each rule. Or should I count the number of m or n conditions?
        :param root: root node of the tree
        :param ret_list: list of the number of nodes between each leaf and the root node
        :return: ret_list updated with the paths to all the leaves
        """
        if ret_list is None:
            ret_list = []
        if self.is_leaf(root):
            ret_list += [root.level]
        if root.left is not None:
            self.leaf_values(root.left, ret_list)
        if root.right is not None:
            self.leaf_values(root.right, ret_list)
        return ret_list
