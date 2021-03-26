import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from cvxopt import solvers, matrix
from scipy.sparse import csr_matrix

# import all data
data_x1 = np.array(pd.read_csv("./data/Xtr0_mat100.csv", sep=" ", header = None))
data_x1_seq = pd.read_csv("./data/Xtr0.csv", sep=",")
data_x1_seq = np.array(data_x1_seq[data_x1_seq.columns[1]].copy())
data_y1 = pd.read_csv("./data/Ytr0.csv", sep=",")

data_x2 = np.array(pd.read_csv("./data/Xtr1_mat100.csv", sep=" ", header = None))
data_x2_seq = pd.read_csv("./data/Xtr1.csv", sep=",")
data_x2_seq = np.array(data_x2_seq[data_x2_seq.columns[1]].copy())
data_y2 = pd.read_csv("./data/Ytr1.csv", sep=",")

data_x3 = np.array(pd.read_csv("./data/Xtr2_mat100.csv", sep=" ", header = None))
data_x3_seq = pd.read_csv("./data/Xtr2.csv", sep=",")
data_x3_seq = np.array(data_x3_seq[data_x3_seq.columns[1]].copy())
data_y3 = pd.read_csv("./data/Ytr2.csv", sep=",")

te_x1 = np.array(np.array(pd.read_csv("./data/Xte0_mat100.csv", sep=" ", header = None)))
te_x2 = np.array(np.array(pd.read_csv("./data/Xte1_mat100.csv", sep=" ", header = None)))
te_x3 = np.array(np.array(pd.read_csv("./data/Xte2_mat100.csv", sep=" ", header = None)))

te_x1_string = pd.read_csv("./data/Xte0.csv", sep=",")
te_x1_string = np.array(te_x1_string[te_x1_string.columns[1]].copy())

te_x2_string = pd.read_csv("./data/Xte1.csv", sep=",")
te_x2_string = np.array(te_x2_string[te_x2_string.columns[1]].copy())

te_x3_string = pd.read_csv("./data/Xte2.csv", sep=",")
te_x3_string = np.array(te_x3_string[te_x3_string.columns[1]].copy())

def split(data_x, data_y):
    x = data_x.copy()
    y = np.array(data_y[data_y.columns[1]].copy())
    #change y to -1 +1
    y = 2*y - 1
    
    #n = 1600
    n = len(x)+1
    validation_x = x[n:].copy()
    validation_y = y[n:].copy()
    x = x[:n]
    y = y[:n]

    return x, y, validation_x, validation_y


x1, y1, validation_x1, validation_y1 = split(data_x1, data_y1)
x2, y2, validation_x2, validation_y2 = split(data_x2, data_y2)
x3, y3, validation_x3, validation_y3 = split(data_x3, data_y3)

x1_seq, y1, validation_x1_seq, validation_y1 = split(data_x1_seq, data_y1)
x2_seq, y2, validation_x2_seq, validation_y2 = split(data_x2_seq, data_y2)
x3_seq, y3, validation_x3_seq, validation_y3 = split(data_x3_seq, data_y3)


def gaussian_kernel(x, y, alpha):
    return np.exp( - alpha * np.linalg.norm(x-y)**2 )

def kernel_ridge_regression(x, y, kernel, l):
    n = len(x)
    
    #compute K
    K = np.zeros((n,n))
    for i in range(0, n):
        for j in range(i, n):
            k_ij = kernel(x[i], x[j])
            K[i][j] = k_ij
            K[j][i] = k_ij

    #\alpha = (K+ lambda I)^-1 y
    alpha = np.linalg.inv(K + l*np.identity(n)) * y
    
    f = lambda y: np.sum([alpha[i] * kernel(x[i], y) for i in range(0, n)])
    return f

def accuracy(f, x, y):
    n = len(x)
    predictions = np.zeros(n)
    for i in range(0, n):
        predictions[i] = np.sign(f(x[i]))
    score = (1/n)*sum([predictions[i] == y[i] for i in range(0, n)])
    return score

# fit with kernel ridge regression
"""
alpha_gauss1 = 6
l1 = 0.05
kernel1 = lambda x, y: gaussian_kernel(x, y, alpha_gauss1)

f1 = kernel_ridge_regression(x1, y1, kernel1, l1)
#print(accuracy(f1, x1, y1))
#print(accuracy(f1, validation_x1, validation_y1))

alpha_gauss2 = 6
l2 = 0.05
kernel2 = lambda x, y: gaussian_kernel(x, y, alpha_gauss2)

f2 = kernel_ridge_regression(x2, y2, kernel2 , l2)
#print(accuracy(f2, x2, y2))
#print(accuracy(f2, validation_x2, validation_y2))

alpha_gauss3 = 6
l3 = 0.05
kernel3 = lambda x, y: gaussian_kernel(x, y, alpha_gauss3)

f3 = kernel_ridge_regression(x3, y3, kernel3, l3)
#print(accuracy(f3, x3, y3))
#print(accuracy(f3, validation_x3, validation_y3))

pred1 = []
for i in te_x1:
    pred1.append(np.sign(f1(i)))
pred2 = []
for i in te_x2:
    pred2.append(np.sign(f2(i)))
pred3 = []
for i in te_x3:
    pred3.append(np.sign(f3(i)))
pred = pred1 + pred2 + pred3
for i in range(len(pred)):
    pred[i] = int((pred[i] + 1)/2)

df = pd.DataFrame({'Bound': pred,})
df.to_csv('Yte.csv', index_label = 'Id')
"""

# SVM

def svm(x, y, kernel, l):
    n=len(x)
        
    K = np.zeros((n,n))
    for i in range(0, n):
        for j in range(i, n):
            k_ij = kernel(x[i], x[j])
            K[i][j] = k_ij
            K[j][i] = k_ij

        
    D = np.diag(y)
    P = matrix( np.array(np.dot(D, np.dot(K, D ))), tc='d')
    q = matrix(-1*np.ones((n, 1)), tc='d')
    g1 = -1*np.eye(n)
    g2 = 1*np.eye(n)
    G = matrix(np.concatenate((g1, g2), axis=0))
    h1 = np.zeros((n, 1))
    h2 = (1/(2*l*n))*np.ones((n, 1))
    h =matrix(np.concatenate((h1, h2), axis=0))
    A = matrix(y.reshape((1, n)), tc='d')
    b = matrix(np.array([0]), tc = 'd')

    sol = solvers.qp(P, q, G, h, A, b)
        
    c = sol['x']
    i = 0
    for j in range(len(c)):
        if c[j] > 10**(-8):
            i = j
            break
    b = sum([c[j]*y[j]*kernel(x[j], x[i]) for j in range(len(c))]) - y[i]
    
    #support_vector = [i for i in range(len(x)) if c[i] > 10**(-6)]
    #c = np.array(c[support_vector])
    #x = np.array(x[support_vector])
    #y = np.array(y[support_vector])
    
    # fonction de classification
    #f = lambda u: sum([c[i]*y[i]*kernel(x[i],u) for i in range(len(c))]) - b
    #return f
    def f(u):
        return sum([c[i]*y[i]*kernel(x[i], u) for i in range(len(c))]) - b
    return f

# fit SVM with gaussian kernel
"""
alpha_gauss1 = 6
l1 = 0.0000005
kernel1 = lambda x, y: gaussian_kernel(x, y, alpha_gauss1)

svm_1 = svm(x1, y1, kernel1, l1)
print(accuracy(svm_1, x1, y1))
print(accuracy(svm_1, validation_x1, validation_y1))

alpha_gauss2 = 3
l2 = 0.000001
kernel2 = lambda x, y: gaussian_kernel(x, y, alpha_gauss2)

svm_2 = svm(x2, y2, kernel2, l2)
print(accuracy(svm_2, x2, y2))
print(accuracy(svm_2, validation_x2, validation_y2))

alpha_gauss3 = 6
l3 = 0.005
kernel3 = lambda x, y: gaussian_kernel(x, y, alpha_gauss3)

svm_3 = svm(x3, y3, kernel3, l3)
print(accuracy(svm_3, x3, y3))
print(accuracy(svm_3, validation_x3, validation_y3))

pred1 = []
for i in te_x1:
    pred1.append(np.sign(svm_1(i)))
pred2 = []
for i in te_x2:
    pred2.append(np.sign(svm_2(i)))
pred3 = []
for i in te_x3:
    pred3.append(np.sign(svm_3(i)))
pred = pred1 + pred2 + pred3
for i in range(len(pred)):
    pred[i] = int((pred[i] + 1)/2)

df = pd.DataFrame({'Bound': pred,})
df.to_csv('Yte.csv', index_label = 'Id')
"""

# K-spectrum

def index(seq):
    binary = []
    for a in seq:
        b = [0,0]
        if a == 'A':
            b = [0,0]
        if a == 'C':
            b = [0,1]
        if a == 'G':
            b = [1,0]
        if a == 'T':
            b = [1,1]
        binary = binary + b
    index = 0
    for digits in binary:
        index = (index << 1) | digits
    return index
        
def spectrum(seq, k=5):
    sub_seq = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    phi_x = np.zeros(4**k)
    for s in sub_seq:
        i = index(s)
        phi_x[i] = phi_x[i] + 1
    return phi_x

def kernel_spectrum(x, y):
    phi_x = spectrum(x)
    phi_y = spectrum(y)
    return np.dot(phi_x, phi_y)


# SVM but just for the kernel spectrum
#(needed to compute \phi(x_i) only once)
def svm_optim(x, y, kernel, l):
    n=len(x)
    
    K = np.zeros((n,n))
    phi = []
    for i in range(n):
        phi.append(spectrum(x[i]))
    phi = np.array(phi)
    phi = csr_matrix(phi).toarray()
    for i in range(0, n):
        for j in range(i, n):
            k_ij = phi[i].dot(phi[j])
            K[i][j] = k_ij
            K[j][i] = k_ij
    #print("rank:")
    #print(np.linalg.matrix_rank(K))
    
    D = np.diag(y)
    P = matrix( np.array(np.dot(D, np.dot(K, D ))), tc='d')
    q = matrix(-1*np.ones((n, 1)), tc='d')
    g1 = -1*np.eye(n)
    g2 = 1*np.eye(n)
    G = matrix(np.concatenate((g1, g2), axis=0))
    h1 = np.zeros((n, 1))
    h2 = (1/(2*l*n))*np.ones((n, 1))
    h =matrix(np.concatenate((h1, h2), axis=0))
    A = matrix(y.reshape((1, n)), tc='d')
    b = matrix(np.array([0]), tc = 'd')

    sol = solvers.qp(P, q, G, h, A, b)
        
    c = sol['x']
    i = 0
    for j in range(len(c)):
        if c[j] > 10**(-8):
            i = j
            break
    b = sum([c[j]*y[j]*kernel(x[j], x[i]) for j in range(len(c))]) - y[i]
    
    support_vector = [i for i in range(len(x)) if c[i] > 10**(-6)]
    c = np.array(c[support_vector])
    x = np.array(x[support_vector])
    y = np.array(y[support_vector])
    phi = np.array(phi[support_vector])

    # fonction de classification
    def f(u):
        phi_u = spectrum(u)
        return sum([c[i]*y[i]*phi[i].dot(phi_u) for i in range(len(c))]) - b
     #f = lambda u: sum([c[i]*y[i]*np.dot(phi[i],spectrum(u)) for i in range(len(c))]) - b
    return f

# fit SVM with spectrum kernel

# 1

l1 = 0.001

svm_1 = svm_optim(x1_seq, y1, kernel_spectrum, l1)


# 2

l2 = 0.03

svm_2 = svm_optim(x2_seq, y2, kernel_spectrum, l2)


# 3

l3 = 0.06

svm_3 = svm_optim(x3_seq, y3, kernel_spectrum, l3)


pred1 = []
for i in te_x1_string:
    pred1.append(np.sign(svm_1(i)))
pred2 = []
for i in te_x2_string:
    pred2.append(np.sign(svm_2(i)))
pred3 = []
for i in te_x3_string:
    pred3.append(np.sign(svm_3(i)))
pred = pred1 + pred2 + pred3
for i in range(len(pred)):
    pred[i] = int((pred[i] + 1)/2)


df = pd.DataFrame({'Bound': pred,})
df.to_csv('Yte.csv', index_label = 'Id')
