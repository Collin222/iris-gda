# n = # features, m = # examples, k = class

import csv
import math

X = []
Y = []
CLASSES=3

Y_encoding = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}

with open('iris.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        features = row[:4]
        for i in range(len(features)):
            features[i] = float(features[i])
        features.insert(0, 1)
        y = row[4]

        X.append(features)
        Y.append(Y_encoding[y])

train_num = 130
X_test = X[train_num:]
Y_test = Y[train_num:]
X = X[:train_num]
Y = Y[:train_num]

m = len(X)
n = 4

print(f"Parsed {m} examples with 4 features, {len(X_test)} examples for testing")

def dot_product(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def h(theta_k, x):
    return dot_product(theta_k, x)

def predict_logits(theta, x):
    logits = []
    for theta_k in theta:
        logits.append(h(theta_k, x))
    return logits

ALPHA = 0.01
def apply_gradient(theta):
    new_theta = []
    for t in theta:
        new_theta.append(list(t))

    for i in range(m):
        x_i = X[i]
        y_i = Y[i]

        logits = predict_logits(theta, x_i)
        normalized = [math.exp(z) for z in logits]

        p = []

        for z in normalized:
            divisor = 0
            for k in range(CLASSES):
                divisor += math.exp(dot_product(theta[k], x_i))
            p.append(z / divisor)

        # true y, one hot encoded
        t = [1 if k == y_i else 0 for k in range(CLASSES)]

        for k in range(CLASSES):
            for j in range(n + 1):
                grad = 1 / m * ALPHA * (p[k] - t[k]) * x_i[j]
                new_theta[k][j] -= grad

    return new_theta

def calc_avg_loss(theta):
    sum = 0
    for i in range(m):
        x_i = X[i]
        y_i = Y[i]
        theta_y = theta[y_i]

        z_k = math.exp(h(theta_y, x_i))

        divisor = 0
        for k in range(CLASSES):
            divisor += math.exp(dot_product(theta[k], x_i))

        p = z_k / divisor

        sum += -1 * math.log(p)
    
    return sum / m

EPOCHS = 10000
def train():
    theta = []
    for _ in range(CLASSES):
        theta.append([0] * (n + 1))

    for e in range(EPOCHS):
        if e % 100 == 0:
            loss = calc_avg_loss(theta)
            print("epoch", e)
            print("loss", loss)

        theta = apply_gradient(theta)

    return theta

theta = train()

correct = 0
for i in range(len(X_test)):
    x_i = X_test[i]
    y_i = Y_test[i]

    max = None
    max_k = -1
    for k in range(CLASSES):
        prediction = h(theta[k], x_i)
        if max is None or prediction > max:
            max = prediction
            max_k = k

    if max_k == y_i:
        correct += 1

accuracy = correct / len(X_test)
print("accuracy", accuracy)
 