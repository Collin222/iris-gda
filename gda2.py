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
        print("v1i", v1[i])
        print("v2i", v2[i])
        sum += v1[i] * v2[i]
    return sum

def subtract_vectors(a, b):
    v = []
    for i in range(len(a)):
        v.append(a[i] - b[i])
    return v

mus = []
covariance_matrix = []

def add_vectors(a, b):
    v = []
    for i in range(len(a)):
        v.append(a[i] + b[i])
    return v

def scale_vector(v, c):
    for i in range(len(v)):
        v[i] *= c
    return v

def fit_mus():
    for k in range(CLASSES):
        total = 0
        sum = [0] * n
        for i in range(m):
            if Y[i] != k:
                continue

            total += 1
            sum = add_vectors(sum, X[i])

        scale_vector(sum, 1/total)
        mus.append(sum)
    return mus

fit_mus()

def scale_matrix(A, c):
    for r in range(len(A)):
        for col in range(len(A[0])):
            A[r][col] *= c
    return A

def add_matricies(A, B):
    C = []
    for r in range(len(A)):
        row = []
        for c in range(len(A[0])):
            row.append(A[r][c] + B[r][c])
        C.append(row)
    return C

def transpose_vector(v):
    A = []
    for i in range(len(v)):
        A.append([v[i]])
    return A

def matrix_multiplication(A, B):
    if len(A[0]) != len(B):
        raise Exception('matrix_multiplication mismatching dimensions')

    C = []
    for r in range(len(A)):
        row = []
        av = A[r]
        for c in range(len(B[0])):
            bv = []
            for i in range(len(B)):
                bv.append(B[i][c])
            row.append(dot_product(av, bv))
        C.append(row)
    return C

def fit_covariance_matrix():
    sum = [] # nxn
    for i in range(n):
        sum.append([0] * n)

    for i in range(m):
        y = Y[i]
        mu = mus[y]
        x = X[i]
        diff = subtract_vectors(x, mu) # 1xn
        transposed = transpose_vector(diff) # nx1
        prod = matrix_multiplication(transposed, [diff]) # nxn
        sum = add_matricies(sum, prod) # nxn

    scale_matrix(sum, 1/m)
    covariance_matrix = sum
    return sum

covariance_matrix = fit_covariance_matrix()

def calc_p_y(y):
    count = 0
    for i in range(m):
        if Y[i] == y:
            count += 1

    return count / m

def calc_norm(A):
    return math.sqrt(sum(x**2 for row in A for x in row))

def matrix_inverse(A):
    n = len(A)
    # Augment A with identity matrix
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] 
         for i, row in enumerate(A)]
    
    for col in range(n):
        # Partial pivoting: find row with max abs value in this column
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular, cannot invert")
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Normalize pivot row
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]
        
        # Eliminate this column from all other rows
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][c] - factor * M[col][c] for c in range(2 * n)]
    
    # Extract right half (the inverse)
    return [row[n:] for row in M]

def calc_p_x_y(x, y):
    mu = mus[y]
    diff = subtract_vectors(x, mu)
    inverse = matrix_inverse(covariance_matrix)
    norm = calc_norm(covariance_matrix)
    frac = 1 / (((2 * math.pi) ** (n / 2)) * (norm ** (1/2)))
    multiplied = matrix_multiplication([diff], inverse)
    print("m", multiplied)
    print("d", diff)
    exponential = math.exp(-1/2 * dot_product(multiplied[0], diff))
    return frac * exponential

def calc_p_x(x):
    sum = 0
    for k in range(CLASSES):
        p_x_y = calc_p_x_y(x, k)
        p_y = calc_p_y(k)
        sum += p_x_y * p_y
    return sum

def calc_p_y_x(x, y):
    p_y = calc_p_y(y)  
    p_x = calc_p_x(x)
    p_x_y = calc_p_x_y(x, y)
    return p_x_y * p_y / p_x

def h(x):
    max = None
    max_class = None
    for k in range(CLASSES):
        prob = calc_p_y_x(x, k)

        if max == None or prob > max:
            max = prob
            max_class = k

    # (probability/confidence in [0, 1] (real), class in [0, 2] (whole))
    return (max, max_class)

def calc_accuracy():
    correct = 0
    test_num = len(X_test)
    prob_sum = 0
    for i in range(test_num):
        x_i = X_test[i]
        y_i = Y_test[i]
        prediction = h(x_i)
        if y_i == prediction[1]:
            correct += 1
        prob_sum += prediction[0]
    return (correct / test_num, prob_sum / test_num)

(accuracy, avg_prob) = calc_accuracy()
print("Accuracy :", accuracy)
print("Average probability :", avg_prob)
