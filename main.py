import csv
import math

X = []
Y = []

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
        y = row[4]

        X.append(features)
        Y.append(Y_encoding[y])

train_num = 130
X_test = X[train_num:]
Y_test = Y[train_num:]
X = X[:train_num]
Y = Y[:train_num]

N = len(Y)
FEATURES = 4

print(f"Parsed {N} examples with 4 features, {len(X_test)} examples for testing")

phis = {}
mus = {0: [0] * 4, 1: [0] * 4, 2: [0] * 4}
covariance = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

occurences = {0: 0, 1: 0, 2: 0}

# calculate phis
for y in Y_encoding.values():
    occurs = 0
    for c in Y:
        if c == y:
            occurs += 1
    
    occurences[y] = occurs
    phi = occurs / N
    phis[y] = phi

# calculate mus
for i in range(len(X)):
    x = X[i]
    y = Y[i]

    for j in range(4):
        mus[y][j] += x[j]

for y in Y_encoding.values():
    for i in range(4):
        mus[y][i] /= occurences[y]

def add_matricies(m1, m2):
    if len(m1) != len(m2):
        raise 'Dimensions do not match'

    if len(m1[0]) != len(m2[0]):
        raise 'Inner dimmensions do not match'
    
    m = []
    for i in range(len(m1)):
        inner = []
        for j in range(len(m1[i])):
            inner.append(m1[i][j] + m2[i][j])

        m.append(inner)

    return m

def subtract_vectors(v1, v2):
    if len(v1) != len(v2):
        raise 'Dimensions do not match'
    
    v = []
    for i in range(len(v1)):
        v.append(v1[i] - v2[i])
    
    return v

def vector_to_matrix(v):
    m = []
    for x in v:
        m.append([x])
    return m

def transpose_matrix(m):
    rows = len(m)
    cols = len(m[0])

    t = []

    for i in range(cols):
        row = []
        for j in range(rows):
            row.append(m[j][i])
        t.append(row)
    
    return t

def dot_product(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def multiply_matricies(m1, m2):
    if len(m1[0]) != len(m2):
        raise 'Matrix multiplication rules violated'
    
    m = []
    
    for r in range(len(m1)):
        row = []
        for c in range(len(m2[0])):
            v1 = m1[r]
            v2 = []
            for j in range(len(m2)):
                v2.append(m2[j][c])

            row.append(dot_product(v1, v2))
        m.append(row)

    return m            

# calculate covariance matrix
for i in range(len(X)):
    x = X[i]
    y = Y[i]

    diff = [subtract_vectors(x, mus[y])]
    transpose = transpose_matrix(diff)
    prod = multiply_matricies(transpose, diff)

    covariance = add_matricies(covariance, prod)

for r in range(len(covariance)):
    for c in range(len(covariance[0])):
        covariance[r][c] /= N

def calc_determinant(m):
    if len(m) == 1:
        return m[0][0]
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    
    sum = 0
    
    for i in range(len(m[0])):
        coefficient = 1 if i % 2 == 0 else -1
        sub_matrix = []
        for r in m[1:]:
            row = []
            for c in range(len(m[0])):
                if c == i:
                    continue
                row.append(r[c])
            sub_matrix.append(row)
        
        sum += coefficient * m[0][i] * calc_determinant(sub_matrix)
    
    return sum

def cofactor_matrix(m):
    cofactors = []

    for r in range(len(m)):
        row = []
        for c in range(len(m[0])):
            sub_matrix = []
            for i in range(len(m)):
                sub_row = []
                if r == i:
                    continue

                for j in range(len(m[i])):
                    if c == j:
                        continue
                    sub_row.append(m[i][j])
                sub_matrix.append(sub_row)
            
            det = calc_determinant(sub_matrix)
            row.append(det * pow(-1, r + c))
        cofactors.append(row)
    return cofactors

def calc_inverse(m):
    det = calc_determinant(m)
    adjugate = transpose_matrix(cofactor_matrix(m))
    inverse = []

    for r in adjugate:
        row = []
        for v in r:
            row.append(v / det)
        inverse.append(row)
    
    return inverse


det_covariance = calc_determinant(covariance)
covariance_inverse = calc_inverse(covariance)

def decode_y(value):
    for x in Y_encoding.items():
        if x[1] == value:
            return x[0]

def predict(x):
    probabilities = []
    for y in Y_encoding.values():
        phi = phis[y]
        diff = [subtract_vectors(x, mus[y])]
        transpose = transpose_matrix(diff)
        prod1 = multiply_matricies(diff, covariance_inverse)
        prod2 = multiply_matricies(prod1, transpose)
        scalar = -0.5 * prod2[0][0]
        X_y = 1 / (pow(2 * math.pi, FEATURES / 2) * pow(det_covariance, 0.5)) * math.exp(scalar)

        probability = phi * X_y
        probabilities.append((y, probability))

    max = None
    for x in probabilities:
        if max is None:
            max = x
        elif x[1] > max[1]:
            max = x

    if max is None:
        raise 'Something went wrong, no prediction calculated'
    
    return max

correct = 0
test_examples = len(X_test)
for i in range(test_examples):
    x = X_test[i]
    y = Y_test[i]

    prediction = predict(x)[0]
    if (prediction == y):
        correct += 1

accuracy = correct / test_examples * 100

print(f"Accuracy: {accuracy:.2f}")
