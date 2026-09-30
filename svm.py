import random
from cvxopt import matrix, solvers
import csv
import math

X = []
Y = []

Y_encoding = {
    "setosa": -1,
    "versicolor": 1,
}

with open('iris.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        features = row[:4]
        for i in range(len(features)):
            features[i] = float(features[i])
        y_raw = row[4]
        if y_raw not in Y_encoding:
            continue
        y = Y_encoding[y_raw]

        X.append(features)
        Y.append(y)

def normalize_x(x, min, max):
    if max - min == 0:
        return x
    return (x - min) / (max - min)

def normalize_examples(examples):
    n = len(examples[0])
    for i in range(n):
        min = None
        max = None
        for j in range(len(examples)):
            xji = examples[j][i]
            if min == None or xji < min:
                min = xji
            if max == None or xji > max:
                max = xji

        for j in range(len(examples)):
            examples[j][i] = normalize_x(examples[j][i], min, max)

normalize_examples(X)

class SVM:
    def __init__(self, X, y, C=1.0):
        test_size = math.floor(0.2 * len(X))

        self.alphas = []
        self.support_alphas = []
        self.b = 0
        self.X = list(X)
        self.y = list(y)
        self.C = C

        self.X_test = []
        self.y_test = []
        for _ in range(test_size):
            idx = random.randint(0, len(self.X) - 1)
            self.X_test.append(self.X[idx])
            self.y_test.append(self.y[idx])
            self.X.pop(idx)
            self.y.pop(idx)

        self.n = len(self.X[0])
        self.m = len(self.X)

        print(f"{self.n} features, {self.m} examples, {test_size} test examples")

        self.fit()

        train_accuracy = self.test_accuracy(self.X, self.y)
        test_accuracy = self.test_accuracy(self.X_test, self.y_test)
        print(f"Train accuracy : {train_accuracy}")
        print(f"Test accuracy : {test_accuracy}")

    def dot_product(self, x, z):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * z[i]
        return sum

    def scale_vector(self, c, a):
        for i in range(len(a)):
            a[i] *= c
        return a

    def linear_kernel(self, x, z):
        return self.dot_product(x, z)

    def rbf_kernel(self, x, z, gamma=0.5):
        sq_dist = sum((a - b) ** 2 for a, b in zip(x, z))
        return math.exp(-gamma * sq_dist)

    def kernel(self, x, z):
        return self.rbf_kernel(x, z)

    def fit(self, tol=1e-5):
        flat_arr = [self.kernel(x=self.X[i], z=self.X[j]) * self.y[i] * self.y[j] for j in range(self.m) for i in range(self.m)]
        P = matrix(size=(self.m, self.m), x=flat_arr)
        q = matrix(size=(self.m, 1), x=[-1.0] * self.m)

        G_flat = [-1.0 if r == j else 1.0 if r == self.m + j else 0.0 for j in range(self.m) for r in range(2*self.m)]
        G = matrix(size=(2*self.m, self.m), x=G_flat)
        h = matrix(size=(2*self.m, 1), x=[0.0] * self.m + [self.C] * self.m)

        A = matrix(size=(1, self.m), x=self.y, tc='d')
        b = matrix(size=(1, 1), x=0.0)

        sol = solvers.qp(P, q, G, h, A, b)
        self.alphas = list(sol['x'])

        for i in range(len(self.alphas)):
            a = self.alphas[i]
            if a > tol:
                self.support_alphas.append([a, i])

        print(f"{len(self.support_alphas)} support vectors")

        b_sum = 0
        on_margin = [a for a in self.support_alphas if a[0] > tol and a[0] < self.C - tol]
        if not on_margin:
            print("None on margin, using support alphas")
            on_margin = self.support_alphas
        for [_, idx] in on_margin:
            ys = self.y[idx]
            pred = 0
            for [a, i] in self.support_alphas:
                pred += a * self.y[i] * self.kernel(self.X[idx], self.X[i])
            b_sum += ys - pred
        self.b = b_sum / len(on_margin)
        print(f"b : {self.b}")

        self.verify_y_linear_comb(tol=tol)

    def verify_y_linear_comb(self, tol):
        sum = 0
        for i in range(self.m):
            sum += self.y[i] * self.alphas[i]
        print(f"y linear comb sum : {sum}")
        if abs(sum) < tol:
            print("linear comb is equal to 0")
        else:
            print("linear comb not eqaul to 0")
        return sum

    def predict(self, x):
        pred = 0
        for [a, idx] in self.support_alphas:
            pred += a * self.y[idx] * self.kernel(x, self.X[idx])
        pred += self.b
        return 1 if pred > 0 else -1

    def test_accuracy(self, X, y):
        correct = 0
        size = len(X)
        for i in range(size):
            xi = X[i]
            yi = y[i]
            pred = self.predict(xi)
            if yi == pred:
                correct += 1

        return (correct / float(size)) * 100


model = SVM(X, y=Y, C=1.0)
