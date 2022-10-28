import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from random import randint
import sys
from datetime import datetime

from tester import extract


def initialisation(couche):
    C = len(couche)
    parametres = {}
    for c in range(1, C):
        parametres["W{}".format(c)] = np.random.randn(couche[c], couche[c - 1])
        parametres["b{}".format(c)] = np.random.randn(couche[c], 1)
    return parametres


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)


def forward_propagation(A0, parametres):
    activations = {"A0": A0}
    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres["W{}".format(c)].dot(activations["A{}".format(c - 1)]) + parametres["b{}".format(c)]
        if c == C:
            activations["A{}".format(c)] = softmax(Z)
        else:
            activations["A{}".format(c)] = 1 / (1 + np.exp(-Z))
    return activations


def back_propagation(reponse_attendue, activations, parametres):
    m = reponse_attendue.shape[1]
    C = len(parametres) // 2
    dZ = activations["A{}".format(C)] - reponse_attendue
    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients["dW{}".format(c)] = np.dot(dZ, activations["A{}".format(c - 1)].T) / m
        gradients["db{}".format(c)] = np.sum(dZ, axis=1, keepdims=True) / m
        if c > 1:
            dZ = np.dot(parametres["W{}".format(c)].T, dZ) * activations["A{}".format(c - 1)] * (
                    1 - activations["A{}".format(c - 1)])
    return gradients


def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    for c in range(1, C + 1):
        parametres["W{}".format(c)] = parametres["W{}".format(c)] - learning_rate * gradients["dW{}".format(c)]
        parametres["b{}".format(c)] = parametres["b{}".format(c)] - learning_rate * gradients["db{}".format(c)]
    return parametres


def log_loss(reponse_attendue, A):
    eps = 1e-15
    return np.sum(-reponse_attendue * np.log(A + eps) - (1 - reponse_attendue) * np.log(1 - A + eps)) / len(y)


def neural_network(features, reponse_attendue, hidden_layers, learning_rate=0.1, n_iter=10000, features_test=None,
                   ytest=None, init=False, poids=None,
                   biais=None, critere=1.0, nbiter=0):
    global accuracy
    np.random.seed(0)
    couche = list(hidden_layers)
    couche = [features.shape[0]] + couche
    # couche.append(y.shape[0])
    if not init:
        parametres = initialisation(couche)
    else:
        parametres = {}
        for indice in range(len(poids)):
            parametres["W{}".format(indice + 1)] = poids[indice]
            parametres["b{}".format(indice + 1)] = biais[indice]

    train_loss, train_acc = [], []
    preclogloss, n_effectif_iter = 0, nbiter
    for _ in tqdm(range(n_iter)):
        activations = forward_propagation(features, parametres)
        gradients = back_propagation(reponse_attendue, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)
        n_effectif_iter += 1

        C = len(parametres) // 2
        logloss = log_loss(reponse_attendue, activations["A{}".format(C)])
        train_loss.append(logloss)
        # print(abs(logloss-preclogloss))
        if abs(logloss - preclogloss) < critere:
            break
        preclogloss = logloss
        y_pred = predict(X, parametres)
        current_accuracy = accuracy_score(reponse_attendue.flatten(), y_pred.flatten())
        train_acc.append(current_accuracy)
        try:
            y_pred = predict(features_test, parametres)
            current_accuracy = accuracy_score(ytest.flatten(), y_pred.flatten())
            test_acc.append(current_accuracy)
        except:
            pass
    accuracy = train_acc[-1]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train loss")
    plt.legend()
    plt.title("{} sous-couches\n{}".format(len(couche) - 2, str(couches[:-1]).replace(", ", "-")[1:-1]))
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train accuracy")
    plt.legend()
    plt.title("{} itérations".format(n_effectif_iter))
    plt.savefig("D:/courbes/2_{}sc{}_{}".format(len(couche) - 1, str(couches[:-1]).replace(", ", "-")[1:-1],
                                                datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
    plt.show()
    return parametres, n_effectif_iter


def predict(features, parametres):
    C = len(parametres) // 2
    return forward_propagation(features, parametres)["A{}".format(C)] >= 0.5


np.set_printoptions(threshold=sys.maxsize)


def save(dataset, parametres, path="modeles"):
    nb_couches = len(couches) + 1
    poids = [parametres["W{}".format(_)] for _ in range(1, nb_couches)]
    biais = [parametres["b{}".format(_)] for _ in range(1, nb_couches)]
    temps = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    nom = "modele{}{}.txt".format(dataset, temps)
    t = open(path + nom, "a")
    t.write("W={}\n\n".format(poids))
    t.write("b={}".format(biais))
    t.close()


def normalise_x(vecteur):
    vecteur = vecteur.T.reshape(-1, vecteur.shape[0]) / vecteur.max()
    return vecteur


def normalise_y(vecteur):
    vecteur = vecteur.T.reshape(1, vecteur.shape[0])
    return vecteur


# #Ma dataset
chemin = "dataset.hdf5"


def pre_process_data(train_y, test_y):
    # Normalize
    # enc = OneHotEncoder(sparse=False, categories='auto')
    # train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    # test_y = enc.transform(test_y.reshape(len(test_y), -1))
    return train_y, test_y


f = h5py.File(chemin, mode='r')
X = np.array(f["train"])
Xtest = np.array(f["test"])
X, Xtest = normalise_x(X), normalise_x(Xtest)
Y = {}
"""
for i in range(10):
    Y["y{}".format(i)], Y["ytest{}".format(i)] = pre_process_data(np.array(f["trainlabel{}".format(i)]),
                                                                  np.array(f["testlabel{}".format(i)]))
                                                                  """
Y["yall"], Y["ytestall"] = pre_process_data(np.array(f["trainlabel"]), np.array(f["testlabel"]))
f.close()

"""
plt.figure(figsize=(16,8))
for i in range(1, 10):
    plt.subplot(4,5,i)
    plt.imshow(X[i], cmap="gray")
    plt.tight_layout()
plt.show()
"""

# #Stratégie 2
# X = X.T.reshape(-1, X.shape[0])/X.max()
# y = y.T.reshape(1, y.shape[0])
accuracy = 0.0
nombreffectifiter = 0
# couches = [256,16,128, 64, 64, 64, 64, 64, 128]
"""
for _ in range(50):
    couches = []
    n = 50
    while n > 0:
        nb = randint(1,n)
        couches.append(nb)
        n-=nb
    W = [np.zeros((couches[0],X.shape[0]))]
    B = [np.zeros((couches[0], 1))]
    for i in range(1, len(couches)):
        w = np.zeros((couches[i], couches[i-1]))
        b = np.zeros((couches[i], 1))
        W.append(w)
        B.append(b)
    W, B = neural_network(features, reponse_attendue, couches, n_iter=1000, learning_rate=0.01, features_test = Xtest, ytest = ytest,init=True,W=W,B=B,critere=1)

    save(str(couches).replace(", ", "-")[1:-1]+"_{}ac_{}it_".format(accuracy.round(3),nombreffectifiter))
"""
"""
couches = [400]*int(3)+[10]
W = [np.zeros((couches[0],X.shape[0]))]
B = [np.zeros((couches[0], 1))]
for j in range(1, len(couches)):
    w = np.zeros((couches[j], couches[j-1]))
    b = np.zeros((couches[j], 1))
    W.append(w)
    B.append(b)
"""

for i in range(1, 11):
    couches = [50 * i, 50 * i, 50 * i, 10]
    # noinspection PyTypeChecker
    W_b, nombreffectifiter = neural_network(X, Y["yall"], couches, n_iter=1000, learning_rate=0.01, features_test=Xtest,
                                            ytest=Y["ytestall"], init=False, poids=W, biais=b, critere=0)
    save("{}".format(str(couches).replace(", ", "-")[1:-1]) + "_{}ac_{}it_".format(accuracy.round(3),
                                                                                   nombreffectifiter), W_b)

"""
couches = [400, 400, 400, 10]
W_b, nombreffectifiter = neural_network(X, Y["yall"], couches, n_iter=100, learning_rate=0.01, features_test = Xtest, ytest = Y["ytestall"],init=False,W=W,B=b,critere=0)
save("{}".format(str(couches).replace(", ", "-")[1:-1])+"_{}ac_{}it_{}_".format(accuracy.round(3),nombreffectifiter,i))
"""


# #Reprise entraînement


def reprise_entrainement(nom):
    path = "modeles/{}.txt".format(nom)
    it = int(path.split("_")[3][:-2])
    layers = [int(nb_neurones) for nb_neurones in path.split("_")[1].split("-")]
    poids, biais = extract(path)

    # noinspection PyTypeChecker
    parametres, nb_total_iter = neural_network(X, Y["yall"], layers, n_iter=1000, learning_rate=0.01,
                                               features_test=Xtest,
                                               ytest=Y["ytestall"], init=True, poids=poids, biais=biais, critere=1e-2,
                                               nbiter=it)
    save(
        "2_{}".format(str(layers).replace(", ", "-")[1:-1]) + "_{}ac_{}it_".format(accuracy.round(3), nb_total_iter),
        parametres)


# #Dataset aléatoire
# X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print("couche de X : {}".format(X.shape))
print("couche de y : {}".format(y.shape))
plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")
plt.show()

for _ in range(1):
    couches = []
    n = 50
    while n > 0:
        nb = randint(1, n)
        couches.append(nb)
        n -= nb
    W = [np.zeros((couches[0], X.shape[0]))]
    B = [np.zeros((couches[0], 1))]
    for i in range(1, len(couches)):
        w = np.zeros((couches[i], couches[i - 1]))
        b = np.zeros((couches[i], 1))
        W.append(w)
        B.append(b)
    W_b = neural_network(X, y, couches, n_iter=1000000, learning_rate=0.01, features_test=None, ytest=np.array([]),
                         init=True,
                         poids=W, biais=B, critere=1e-4)
