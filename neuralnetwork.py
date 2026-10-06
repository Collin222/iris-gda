# neural network from scratch

import math
import random
import csv

random.seed(0)


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

data = list(zip(X, y))
random.shuffle(data)
X = [x for x, _ in data]
y = [label for _, label in data]

train_num = 130
X_test = X[train_num:]
y_test = y[train_num:]
X = X[:train_num]   
y = y[:train_num]

print(f"Parsed {len(X)} examples with {len(X[0])} features, {len(X_test)} examples for testing")

class NeuralNetwork:
    def __init__(self, layers, X, y, lr=0.01):
        self.lows = None
        self.highs = None
        self.fit_normalization(X)
        self.X = self.normalize(X)
        self.y = y
        self.weights = [] # matrix for each layer
        self.bias = [] # vector for each layer
        self.BATCH_SIZE = 50
        self.lr = lr

        self.m = len(X)
        if self.m < 1:
            raise Exception('no examples to train')
        self.n = len(X[0])

        if len(layers) < 2:
            raise Exception('invalid len of layers')
    
        self.layers = layers
        self.num_layers = len(layers)

        self.initialize()

    def fit_normalization(self, X):
        cols = list(zip(*X))
        self.lows = [min(c) for c in cols]
        self.highs = [max(c) for c in cols]

    def normalize_x(self, x, lo, hi):
        return (x - lo) / (hi - lo + 1e-16)

    def normalize(self, X):
        return [[self.normalize_x(v, lo, hi) for v, lo, hi in zip(row, self.lows, self.highs)] for row in X]

    def initialize(self):
        for i in range(self.num_layers):
            prior_n = self.n if i == 0 else self.layers[i - 1]
            neurons = self.layers[i]
            std = math.sqrt(2 / prior_n)
            w = [[random.gauss(0, std) for _ in range(prior_n)] for _ in range(neurons)]
            self.weights.append(w)
            self.bias.append([0] * neurons)

    def dot_product(self, a, b):
        return sum([a[i] * b[i] for i in range(len(a))])

    def matrix_multiplication(self, A, B):
        C = []
        for r in range(len(A)):
            new_row = []
            for c in range(len(B[0])):
                col = [B[i][c] for i in range(len(B))]
                new_row.append(self.dot_product(A[r], col))
            C.append(new_row)
        return C

    def make_batch(self):
        picked = set()
        batch = []
        i = 0
        MAX_ITER = 1_000
        while len(picked) < self.BATCH_SIZE:
            i += 1
            idx = random.randrange(self.m)
            if i >= MAX_ITER:
                raise Exception('max iter reached making batch')
            
            if idx in picked:
                continue
            item = (self.X[idx], self.y[idx])
            picked.add(idx)
            batch.append(item)
        return batch

    def transpose(self, A):
        rows, cols = len(A), len(A[0])
        return [[A[r][c] for r in range(rows)] for c in range(cols)]

    def broadcast_vector(self, b, times):
        B = [b for _ in range(times)]
        return self.transpose(B)

    def add_matricies(self, A, B):
        return [[A[r][c] + B[r][c] for c in range(len(A[r]))] for r in range(len(A))]
    
    def subtract_matricies(self, A, B):
        return [[A[r][c] - B[r][c] for c in range(len(A[r]))] for r in range(len(A))]

    def add_vectors(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def subtract_vectors(self, a, b):
        return [a[i] - b[i] for i in range(len(a))]

    def scale_matrix(self, s, A):
        return [[A[r][c] * s for c in range(len(A[r]))] for r in range(len(A))]

    def scale_vector(self, s, v):
        return [v[i] * s for i in range(len(v))]

    def relu(self, z):
        return 0 if z <= 0 else z

    def softmax(self, z):
        a = []
        for j in range(len(z[0])):
            # linear part for each neuron of jth example
            zj = [z[i][j] for i in range(len(z))]
            zmax = max(zj)
            zj = [zx - zmax for zx in zj]

            logits = [math.exp(zj[i]) for i in range(len(zj))]
            divisor = sum(logits)

            aj = [logits[i] / divisor for i in range(len(zj))]
            a.append(aj)
        return self.transpose(a)

    def activation(self, z, l):
        # softmax for output layer
        # to create prob. dist. among classes
        if l == self.num_layers - 1:
            return self.softmax(z)
        return [[self.relu(z[i][j]) for j in range(len(z[i]))] for i in range(len(z))]

    def compute_layer(self, weights, b, prior, l):
        # n = # features / neurons in prior layer
        # m = size of batch in current forward prop iteration
        # neurons = # neurons in current layer we are computing

        # weights: neurons x n
        # prior: n x m
        # b: vector of size neurons
        # z: neurons x m

        m = len(prior[0]) # size of batch
        prod = self.matrix_multiplication(weights, prior) # neurons x m
        B = self.broadcast_vector(b, m) # b is neurons x 1, B broadcasted is neurons x m
        z = self.add_matricies(prod, B) # neurons x m
        a = self.activation(z, l) # neurons x m

        return z, a

    def forward_prop(self, batch=None):
        outputs = [] # (z, a)[]

        # n x m
        batch = self.make_batch() if batch is None else batch
        for l in range(self.num_layers):
            weights = self.weights[l] # neurons x n
            bias = self.bias[l] # neurons x 1, vector

            prior = None
            if l == 0:
                # create from batch of inputs
                prior = [x for (x, _) in batch]
                prior = self.transpose(prior)
            else:
                prior = outputs[l - 1][1]

            # both neurons x m
            z, a = self.compute_layer(weights, bias, prior, l)
            
            outputs.append((z, a))

        return outputs, batch

    def compute_prediction(self, logits):
        max_class = None
        max_logit = None
        for k in range(len(logits)):
            z = logits[k]
            if max_logit is None or z > max_logit:
                max_logit = z
                max_class = k
        # probability, class
        return max_logit, max_class

    def predict(self, x):
        [normalized] = self.normalize([x])
        # y is irrelivant in this fn call
        outputs, _ = self.forward_prop([(normalized, 0)])
        a = outputs[len(outputs) - 1][1]
        logits = [z for [z] in a]
        return self.compute_prediction(logits)

    def dimensions(self, a):
        if isinstance(a[0], list):
            return len(a), len(a[0])
        else:
            return len(a), 1

    def backward_prop(self, outputs, batch):
        m = len(batch) # batch size
        batch_x = [batch[r][0] for r in range(m)]

        new_weights = [None] * self.num_layers
        new_bias = [None] * self.num_layers

        gammas = []
        for l in range(self.num_layers - 1, -1, -1):
            weights = self.weights[l] # neurons x n
            bias = self.bias[l] # neurons x 1, vector
            zl, za = outputs[l] # both neurons x m
            a_prior = self.transpose(batch_x) if l == 0 else outputs[l-1][1] # n x m

            gamma = None
            is_output = l == self.num_layers - 1
            if is_output:
                # true y
                y = [[1 if batch[c][1] == r else 0 for c in range(m)] for r in range(len(za))] # neurons x m
                gamma = self.subtract_matricies(za, y) # neurons x m
            else:
                w_next = self.weights[l + 1] # neurons(next) x neurons
                gamma_next = gammas[self.num_layers - l - 2] # neurons(next) x m
                gamma = self.matrix_multiplication(self.transpose(w_next), gamma_next) # neurons x m
                gamma = [[gamma[r][c] if zl[r][c] > 0 else 0 for c in range(len(gamma[r]))] for r in range(len(gamma))] # neurons x m

            gammas.append(gamma)

            w_grad = self.matrix_multiplication(gamma, self.transpose(a_prior)) # neurons x n
            w_grad = self.scale_matrix(1/m*self.lr, w_grad) # average loss

            # apply gradient to weights
            weights = self.subtract_matricies(weights, w_grad)
            new_weights[l] = weights

            # average each neuron grad among the batch
            b_grad = [sum([gamma[r][c] for c in range(len(gamma[r]))]) / len(gamma[r]) for r in range(len(gamma))]

            # apply gradient to bias
            b_grad = self.scale_vector(self.lr, b_grad)
            bias = self.subtract_vectors(bias, b_grad)
            new_bias[l] = bias

        self.weights = new_weights
        self.bias = new_bias

    def compute_avg_loss(self, batch):
        sum = 0
        outputs, _ = self.forward_prop(batch)
        _, a = outputs[len(outputs) - 1] # neurons x m
        classes = len(a) # output neurons = # classes, each with a prob
        for i in range(len(batch)):
            yi = batch[i][1]
            isum = 0
            for k in range(classes):
                yk = 1 if yi == k else 0
                isum += yk * math.log(a[k][i] + 1e-16)
            sum += isum
        return -sum / len(batch)

    def compute_accuracy(self, batch):
        correct = 0
        prob_sum = 0
        outputs, batch = self.forward_prop(batch)
        z, a = outputs[len(outputs) - 1] # neurons x m
        classes = len(a) # output neurons = # classes, each with a prob
        for i in range(len(batch)):
            yi = batch[i][1]
            prob, pred = self.compute_prediction([a[k][i] for k in range(classes)])
            if pred == yi:
                correct += 1
            prob_sum += prob
        return correct / len(batch), prob_sum / len(batch)


    def iteration(self, i):
        if i % 50 == 0:
            demo_batch = self.make_batch()
            avg_loss = self.compute_avg_loss(demo_batch)
            accuracy, avg_prob = self.compute_accuracy(demo_batch)
            print(f"Training iteration {i} with average loss {avg_loss}, accuracy {accuracy}, average probability {avg_prob}")

        outputs, batch = self.forward_prop()
        self.backward_prop(outputs, batch)

    def train(self, epochs=5000):
        for e in range(epochs):
            self.iteration(e)

        print(f"Done training after {epochs} epochs")

nn = NeuralNetwork([5, 3], X, y)
nn.train()
train_accuracy, _ = nn.compute_accuracy(list(zip(nn.X, nn.y)))
test_accuracy, _ = nn.compute_accuracy(list(zip(nn.normalize(X_test), y_test)))
print(f"Train accuracy : {train_accuracy}")
print(f"Test accuracy : {test_accuracy}")
