# decision trees classification
# with bagging and random forests

import csv
import math
import random

X = []
y = []

classes_encoding = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}
CLASSES = list(classes_encoding.values())

with open('iris.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        features = row[:4]
        for i in range(len(features)):
            features[i] = float(features[i])
        y_raw = row[4]

        X.append(features)
        y.append(classes_encoding[y_raw])

random.seed(0)
data = list(zip(X, y))
random.shuffle(data)
X = [x for x, _ in data]
y = [label for _, label in data]

train_num = 130
X_test = X[train_num:]
y_test = y[train_num:]
X = X[:train_num]
y = y[:train_num]

m = len(y) # num train examples
n = 4 # num features

print(f"Parsed {m} examples with {n} features, {len(X_test)} examples for testing")

class Node:
    def __init__(self, j, t, d, children, max_class):
        self.j = j
        self.t = t
        self.d = d
        self.children = children
        self.max_class = max_class

class DecisionTree:
    def __init__(self, X, y, max_depth=5, num_features=2):
        self.max_depth = max_depth
        self.num_features = num_features
        self.features = []

        self.root = self.split(X, y)

        # train_accuracy = self.compute_accuracy(X, y)

        # print(f"Train accuracy : {train_accuracy}")
        # print(f"Test accuracy : {test_accuracy}")

    def is_picked(self, j):
        return j in set(self.features)

    def pick_features(self):
        self.features = []
        remaining = list(range(n))
        for _ in range(self.num_features):
            i = random.randrange(len(remaining))
            self.features.append(remaining[i])
            remaining.pop(i)

    def cross_loss(self, data):
        class_counts = {0: 0, 1: 0, 2: 0}
        for c in data:
            class_counts[c] += 1

        sum = 0
        for c in CLASSES:
            # proportion of class in data
            p = class_counts[c] / len(data)
            if p != 0:
                sum += p * math.log2(p)
        return -1 * sum

    def compute_threshold(self, examples, targets):
        eval_thresholds = [set() for _ in range(n)]

        for j in self.features:
            vals = sorted(set(x[j] for x in examples))
            for i in range(len(vals)):
                if i == len(vals) - 1:
                    continue

                a = vals[i]
                b = vals[i + 1]

                mid = (a + b) / 2
                eval_thresholds[j].add(mid)

        best_threshold = None
        min_loss = None
        best_j = None

        for j in self.features:
            for t in eval_thresholds[j]:
                # splits of classes
                s1 = [] # xj < t
                s2 = [] # xj >= t
                for i in range(len(examples)):
                    x = examples[i]
                    y = targets[i]
                    if x[j] < t:
                        s1.append(y)
                    else:
                        s2.append(y)
                s1_loss = 0 if len(s1) == 0 else self.cross_loss(s1)
                s2_loss = 0 if len(s2) == 0 else self.cross_loss(s2)
                loss = (len(s1) * s1_loss + len(s2) * s2_loss) / len(examples)

                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    best_threshold = t
                    best_j = j

        if best_threshold is None or min_loss is None or best_j is None:
            # identical examples, no midpoints found
            return -1, -1

        return (best_j, best_threshold)

    def is_one_class(self, set):
        for i in range(len(set)):
            if i == len(set) - 1:
                continue

            if set[i] != set[i + 1]:
                return False
        return True

    def split(self, examples, targets, d=0, retry_attempt=0):
        if d >= self.max_depth:
            return None

        if len(examples) == 0:
            return None
        
        if self.is_one_class(targets):
            return Node(0, 0, d, (None, None), targets[0])

        if retry_attempt >= 3:
            self.features = list(range(n)) # use all features
        else:
            self.pick_features()

        print(f"Running split at depth={d} with {len(examples)} examples and {len(self.features)} features")

        class_counts = {0: 0, 1: 0, 2: 0}
        for c in targets:
            class_counts[c] += 1

        max_c = None
        max_count = None
        for c, count in class_counts.items():
            if max_count is None or count > max_count:
                max_c = c
                max_count = count

        if max_c is None:
            raise Exception('failed to compute max class')

        (j, t) = self.compute_threshold(examples, targets)
        if j == -1 and t == -1:
            if retry_attempt >= 3:
                return Node(0, 0, d, (None, None), max_c)
            return self.split(examples, targets, d, retry_attempt + 1)
        
        s1_examples = []
        s1_targets = []
        s2_examples = []
        s2_targets = []
        for i in range(len(examples)):
            x = examples[i]
            y = targets[i]
            if x[j] < t:
                s1_examples.append(x)
                s1_targets.append(y)
            else:
                s2_examples.append(x)
                s2_targets.append(y)

        left = self.split(s1_examples, s1_targets, d + 1)
        right = self.split(s2_examples, s2_targets, d + 1)

        node = Node(j, t, d, (left, right), max_c)
        return node

    def predict(self, x, node=None):
        if node is None:
            node = self.root

        if x[node.j] < node.t:
            if node.children[0] is None:
                return node.max_class
            return self.predict(x, node.children[0])
        else:
            if node.children[1] is None:
                return node.max_class
            return self.predict(x, node.children[1])

  
class Bagging:
    def __init__(self, X, y, X_test, y_test, B=50, p=0.67):
        self.trees = []
        self.X = X
        self.y = y
        self.B = B
        self.p = p

        self.make_trees()

        train_accuracy, train_avg_percent = self.compute_accuracy(X, y)
        test_accuracy, test_avg_percent = self.compute_accuracy(X_test, y_test)

        print(f"Train accuracy : {train_accuracy} with {train_avg_percent} / {self.B} trees agreeing on average")
        print(f"Test accuracy : {test_accuracy} with {test_avg_percent} / {self.B} trees agreeing on average")

    def make_trees(self):
        sample_size = math.floor(self.p * len(self.X))
        print(f"Each bootstrap sample size : {sample_size}")
        for _ in range(self.B):
            data = list(zip(self.X, self.y))
            random.shuffle(data)
            picked = []
            for _ in range(sample_size):
                idx = random.randrange(len(data))
                picked.append(data[idx])
            train_x = [entry[0] for entry in picked]
            train_y = [entry[1] for entry in picked]
            self.trees.append(DecisionTree(train_x, train_y))
        print(f"Made {self.B} trees")

    def predict(self, x):
        class_counts = {0: 0, 1: 0, 2: 0}
        for tree in self.trees:
            pred = tree.predict(x)
            class_counts[pred] += 1

        max_c = None
        max_count = None
        for c, count in class_counts.items():
            if max_count is None or count > max_count:
                max_c = c
                max_count = count

        if max_c is None:
            raise Exception("failed to compute most common class in predict()")
    
        return max_c, max_count

    def compute_accuracy(self, X, y):
        correct = 0
        sum = 0
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            pred, count = self.predict(xi)
            if pred == yi:
                correct += 1
            sum += count
        return correct / len(X), sum / len(X)

Bagging(X, y, X_test, y_test)    
