import numpy as np
import cvxopt.solvers
import data_encoding_sample as de
from data_encoding_sample import DataEncoding
from scipy.linalg import sqrtm
import pd_appox as pd
import argparse

parser = argparse.ArgumentParser()

## hyperparameter

parser.add_argument('--N_TRAIN', type=int, default=100)
parser.add_argument('--N_TEST', type=int, default=100)
parser.add_argument('--SHOTS', type=int, default=1000)

## depolarizing probability
parser.add_argument('--prob1', type=float, default=0.3)
parser.add_argument('--prob2', type=float, default=0.3)

## Number of simulation
parser.add_argument('--sim_num', type=int, default=10)

args = parser.parse_args()


dataencoding = DataEncoding(args)

def get_args():
    return args


data_x_train = np.loadtxt("x_train.txt")
data_x_test = np.loadtxt("x_test.txt")
data_y_train = np.loadtxt("y_train.txt")
data_y_test = np.loadtxt("y_test.txt")

cvxopt.solvers.options['show_progress'] = False


class SVM():
    def __init__(self, kernel="rbf", polyconst=1, gamma=10, degree=2):
        self.kernel = kernel
        self.polyconst = float(1)
        self.gamma = float(gamma)
        self.degree = degree
        self.kf = {
            "linear": self.linear,
            "rbf": self.rbf,
            "poly": self.polynomial,
            "qk": self.qk
        }
        self._support_vectors = None
        self._alphas = None
        self.intercept = None
        self._n_support = None
        self.weights = None
        self._support_labels = None
        self._indices = None

    def linear(self, x, y):
        return np.dot(x.T, y)

    def polynomial(self, x, y):
        return (np.dot(x.T, y) + self.polyconst) ** self.degree

    def rbf(self, x, y):
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, y).T, np.subtract(x, y)))

    def qk(self, x, y, phi_x, phi_z):
        return sum(np.sum(dataencoding.myqnode(x, y, phi_x, phi_z), axis=0) == 2) / args.SHOTS

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                if self.kernel == 'qk':
                    K[i, j] = qk_matrix[i, j]  # self.qk(X[i, :], X[j, :], phi_x, phi_z)
                else:
                    K[i, j] = self.kf[self.kernel](X[i, :], X[j, :])
                K[j, i] = K[i, j]
        return K

    def fit(self, data, labels):
        num_data, num_features = data.shape
        labels = labels.astype(np.double)
        K = self.transform(data)
        if pd.isPD(K):
            K = K
        else:
            K = pd.nearestPD(K)
        P = cvxopt.matrix(np.outer(labels, labels) * K)
        q = cvxopt.matrix(np.ones(num_data) * -1)
        A = cvxopt.matrix(labels, (1, num_data))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(num_data) * -1))
        h = cvxopt.matrix(np.zeros(num_data))

        alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        is_sv = alphas > 1e-5
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = alphas[is_sv]
        self._support_labels = labels[is_sv]
        self._indices = np.arange(num_data)[is_sv]
        self.intercept = 0
        for i in range(self._alphas.shape[0]):
            self.intercept += self._support_labels[i]
            self.intercept -= np.sum(self._alphas * self._support_labels * K[self._indices[i], is_sv])
        self.intercept /= self._alphas.shape[0]
        self.weights = np.sum(data * labels.reshape(num_data, 1) * self._alphas.reshape(num_data, 1), axis=0,
                              keepdims=True) if self.kernel == "linear" else None

    def signum(self, X):
        return np.where(X > 0, 1, -1)

    def project(self, X):
        if self.kernel == "linear":
            score = np.dot(X, self.weights) + self.intercept
        else:
            score = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, label, sv in zip(self._alphas, self._support_labels, self._support_vectors):
                    phix = de.phi_calcu(X[i])
                    phiz = de.phi_calcu(sv)
                    if self.kernel == 'qk':
                        s += alpha * label * self.qk(X[i], sv, phix, phiz)
                    else:
                        s += alpha * label * self.kf[self.kernel](X[i], sv)
                score[i] = s
            score = score + self.intercept
        return score

    def predict(self, X):
        return self.signum(self.project(X))


# relabel based on quantum kernel


qk_train_accuracy = np.zeros(args.sim_num)
qk_test_accuracy = np.zeros(args.sim_num)
qk_num_sv = np.zeros(args.sim_num)
qk_intercept = np.zeros(args.sim_num)
rbf_train_accuracy = np.zeros(args.sim_num)
rbf_test_accuracy = np.zeros(args.sim_num)
rbf_num_sv = np.zeros(args.sim_num)
rbf_intercept = np.zeros(args.sim_num)
Max_class_classifier_training_set_accuracy = np.zeros(args.sim_num)
# q = 1
for q in range(args.sim_num):
    np.random.seed(1 * 10 + 100)
    permutation_train = np.random.permutation(data_x_train.shape[0])
    permutation_test = np.random.permutation(data_x_test.shape[0])
    print("test")
    x_train, x_test = data_x_train[permutation_train, :], data_x_test[permutation_test, :]
    y_train, y_test = data_y_train[permutation_train], data_y_test[permutation_test]
    x_train, x_test = x_train[:args.N_TRAIN], x_test[:args.N_TEST]
    y_train, y_test = y_train[:args.N_TRAIN], y_test[:args.N_TEST]
    print("New number of training examples:", len(x_train))
    print("New number of test examples:", len(x_test))
    print("number of shots:", args.SHOTS)
    print("depolarizing error", args.prob1)
    data = np.concatenate((x_train, x_test), axis=0)

    # model_qk = SVM(kernel="qk", gamma=3)
    model_rbf = SVM(kernel="rbf", gamma=3)

    qk_matrix = dataencoding.quantum_kernel_matrix(data)
    rbf_matrix = model_rbf.transform(data)

    sqrt_qk = sqrtm(qk_matrix)
    inv_rbf = np.linalg.inv(rbf_matrix)
    C_Q_matrix = np.dot(np.dot(sqrt_qk, inv_rbf), sqrt_qk)
    eigvec, eigval, eigvec_t = np.linalg.svd(C_Q_matrix)
    v = eigvec[:, 1].real
    y_construct = np.dot(sqrt_qk, v)
    y_construct = y_construct.real
    y_labels = np.ones_like(y_construct) * (-1)
    y_labels[y_construct > np.median(y_construct)] = 1
    train_y = y_labels[:args.N_TRAIN]
    test_y = y_labels[args.N_TRAIN:]

    train_data = x_train
    test_data = x_test
    N = train_data.shape[0]
    train_labels = train_y.astype(np.double)
    test_labels = test_y.astype(np.double)
    predictions = np.ones_like(train_labels) * -1
    print("Max-class classifier training set accuracy: ", np.mean(np.equal(predictions, train_labels)) * 100, "%")
    model_qk = SVM(kernel="qk", gamma=3)
    model_rbf = SVM(kernel="rbf", gamma=3)
    model_qk.fit(train_data, train_labels)
    model_rbf.fit(train_data, train_labels)
    predictions_qk = model_qk.predict(train_data)
    predictions_rbf = model_rbf.predict(train_data)
    test_predictions_qk = model_qk.predict(test_data)
    test_predictions_rbf = model_rbf.predict(test_data)
    print("%d-th SVM model Training set accuracy for qk: " % q, np.mean(np.equal(predictions_qk, train_labels)) * 100,
          "%")
    print("%d-th SVM model Training set accuracy for rbf: " % q, np.mean(np.equal(predictions_rbf, train_labels)) * 100,
          "%")
    print("%d-th SVM model Testing set accuracy for qk: " % q,
          np.mean(np.equal(test_predictions_qk, test_labels)) * 100,
          "%")
    print("%d-th SVM model Testing set accuracy for rbf: " % q,
          np.mean(np.equal(test_predictions_rbf, test_labels)) * 100,
          "%")
    print("%d-th Number of SVMs computed for qk: " % q, model_qk._n_support)
    print("%d-th Number of SVMs computed for rbf: " % q, model_rbf._n_support)
    print("%d-th Value of intercept for qk: " % q, model_qk.intercept)
    print("%d-th Value of intercept for rbf: " % q, model_rbf.intercept)
    Max_class_classifier_training_set_accuracy[q] = np.mean(np.equal(predictions, train_labels)) * 100
    qk_train_accuracy[q] = np.mean(np.equal(predictions_qk, train_labels)) * 100
    rbf_train_accuracy[q] = np.mean(np.equal(predictions_rbf, train_labels)) * 100
    qk_test_accuracy[q] = np.mean(np.equal(test_predictions_qk, test_labels)) * 100
    rbf_test_accuracy[q] = np.mean(np.equal(test_predictions_rbf, test_labels)) * 100
    qk_num_sv[q] = model_qk._n_support
    rbf_num_sv[q] = model_rbf._n_support
    qk_intercept[q] = model_qk.intercept
    rbf_intercept[q] = model_rbf.intercept

np.savetxt('Max_class_classifier_training_set_accuracy_N' + str(args.N_TRAIN) + 'shots'
           + str(args.SHOTS), Max_class_classifier_training_set_accuracy)
np.savetxt('qk_train_accuracy_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), qk_train_accuracy)
np.savetxt('rbf_train_accuracy_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), rbf_train_accuracy)
np.savetxt('qk_test_accuracy_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), qk_test_accuracy)
np.savetxt('rbf_test_accuracy_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), rbf_test_accuracy)
np.savetxt('qk_num_sv_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), qk_num_sv)
np.savetxt('rbf_num_sv_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), rbf_num_sv)
np.savetxt('qk_intercept_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), qk_intercept)
np.savetxt('rbf_intercept_N' + str(args.N_TRAIN) + 'shots' + str(args.SHOTS), rbf_intercept)

'''if __name__ == '__main__':
    main()
'''
