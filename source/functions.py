from random import randint, choices
from PIL import Image, ImageDraw
import numpy as np
import csv
from tqdm import tqdm
import h5py
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Variables et constantes
nombre_chaque = [0 for _ in range(999999)]  # Compte le nombre captchas de chaque nombre
tailles = ((15, 17),  # 10
           (8, 17),  # 11
           (14, 18),  # 12
           (15, 17),  # 13
           (14, 17),  # 14
           (16, 17),  # 15
           (15, 17),  # 16
           (13, 17),  # 17
           (16, 17),  # 18
           (13, 17),  # 19
           (10, 19),  # 20
           (7, 19),  # 21
           (9, 19),  # 22
           (10, 19),  # 23
           (11, 19),  # 24
           (10, 19),  # 25
           (10, 19),  # 26
           (8, 19),  # 27
           (10, 19),  # 28
           (10, 19))  # 29
path_annotations = "source/annotations.csv"
path_pictures = "creation/{}.png"
path_dataset = "source/dataset.hdf5"

accuracy = 0.0
nombreffectifiter = 0

chiffre10 = Image.open("pixil/10.png")
chiffre11 = Image.open("pixil/11.png")
chiffre12 = Image.open("pixil/12.png")
chiffre13 = Image.open("pixil/13.png")
chiffre14 = Image.open("pixil/14.png")
chiffre15 = Image.open("pixil/15.png")
chiffre16 = Image.open("pixil/16.png")
chiffre17 = Image.open("pixil/17.png")
chiffre18 = Image.open("pixil/18.png")
chiffre19 = Image.open("pixil/19.png")
chiffre20 = Image.open("pixil/20.png")
chiffre21 = Image.open("pixil/21.png")
chiffre22 = Image.open("pixil/22.png")
chiffre23 = Image.open("pixil/23.png")
chiffre24 = Image.open("pixil/24.png")
chiffre25 = Image.open("pixil/25.png")
chiffre26 = Image.open("pixil/26.png")
chiffre27 = Image.open("pixil/27.png")
chiffre28 = Image.open("pixil/28.png")
chiffre29 = Image.open("pixil/29.png")

Liste_chiffres = (
    (chiffre10, chiffre20),
    (chiffre11, chiffre21),
    (chiffre12, chiffre22),
    (chiffre13, chiffre23),
    (chiffre14, chiffre24),
    (chiffre15, chiffre25),
    (chiffre16, chiffre26),
    (chiffre17, chiffre27),
    (chiffre18, chiffre28),
    (chiffre19, chiffre29))


# Creation


def change_couleur_chiffre(image, r_chiffre, g_chiffre, b_chiffre):
    """Change la couleur du chiffre à coller dans le captcha."""
    data = np.array(image)
    red, green, blue, alpha = data.T
    fond = (red == 0) & (green == 0) & (blue == 0)
    data[..., :-1][fond.T] = (r_chiffre, g_chiffre, b_chiffre)
    return Image.fromarray(data)


def appends(l, *args):
    for e in args:
        l.append(e)


def creation(number, show=False):
    number = str(number)
    r, g, b = randint(20, 173), randint(20, 173), randint(20, 173)
    a = choices([200, 255], weights=[2355, 1])[0]
    nombre_de_chiffres = len(
        number)  # Détermine le nombre de chiffres du captcha entre 4 et 6 selon la place disponible
    w = randint(max(118, 10 + 20 * nombre_de_chiffres), 189)
    h = randint(40, 59)

    new_captcha = Image.new("RGBA", (w, h), (r, g, b, a))  # Crée le fond
    # new_captcha.show() #Affiche la base du captcha (fond)

    ligne_csv = []
    r_chiffre, g_chiffre, b_chiffre = r, g, b
    while (r_chiffre, g_chiffre, b_chiffre) == (
            r, g, b):  # S'assure que les chiffres n'aient pas la même couleur que le fond
        r_chiffre, g_chiffre, b_chiffre = randint(20, 173), randint(20, 173), randint(20, 173)

    espace_en_plus = int((w - 2 * 5 - 20 * nombre_de_chiffres) / (
            2 * nombre_de_chiffres - 2))  # Marge en plus en abscisse qu'un chiffre peut prendre

    longueur_precedente, x = 0, 0
    chiffre, y = 0, 0
    for i in range(nombre_de_chiffres):
        chiffre = choices(Liste_chiffres[int(number[i])])[0]  # Choisit une police aléatoire
        chiffre = change_couleur_chiffre(chiffre, r_chiffre, g_chiffre, b_chiffre)  # Change la couleur du chiffre
        longueur, hauteur = chiffre.size
        x = randint((i > 0) * (x + longueur_precedente + 2) + (i == 0) * 5,
                    (0 < i < nombre_de_chiffres - 1) * (5 + 20 * i + espace_en_plus * 2 * (i + 1)) + (
                            i == nombre_de_chiffres - 1) * (w - 5 - longueur_precedente) + (i == 0) * (
                            25 - longueur))
        y = randint(5, h - 5 - hauteur)
        appends(ligne_csv, x - 2, y - 2, x + longueur + 2, y + hauteur + 2)
        longueur_precedente = longueur
        new_captcha.paste(chiffre, (x, y), chiffre)

    new_captcha.paste(chiffre, (x, y), chiffre)

    nb_lignes = randint(4, 5) + 2
    draw = ImageDraw.Draw(new_captcha)

    for i in range(nb_lignes):
        x, y, xp, yp = randint(0, w), randint(0, h), randint(0, w), randint(0, h)  # Extrémités du segment
        epaisseur = randint(0, i // 2)  # Épaisseur de la ligne
        r_ligne, g_ligne, b_ligne = randint(20, 173), randint(20, 173), randint(20, 173)
        draw.line((x, y, xp, yp), fill=(r_ligne, g_ligne, b_ligne), width=epaisseur)  # Dessine la ligne

    x, y = randint(0, w - 50), randint(0, h - 20)
    xp, yp = randint(x, w), randint(y, h)
    angle_debut = randint(0, 89)
    angle_fin = 180 - angle_debut
    epaisseur = randint(1, 3)
    r_arc, g_arc, b_arc = randint(20, 173), randint(20, 173), randint(20, 173)
    draw.arc((x, y, xp, yp), angle_debut, angle_fin, fill=(r_arc, g_arc, b_arc), width=epaisseur)  # Dessine l'arc

    new_captcha.save("creation/{}#{}.png".format(number, nombre_chaque[int(number)]))
    if show:
        new_captcha = Image.open("creation/{}#{}.png".format(number, nombre_chaque[int(number)]))
        new_captcha.show()

    with open("source/annotations.csv", 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["#{}#{}".format(number, nombre_chaque[int(number)])] + ligne_csv)
        f.close()
    nombre_chaque[int(number)] += 1


def captcha_factory(nb_chiffres, quantite):
    for _ in range(quantite):
        nombre = randint(0, int(nb_chiffres * "9"))
        nombre = "0" * (nb_chiffres - len(str(nombre))) + str(nombre)
        try:
            creation(nombre)
            if not int(nombre[-1]):
                print(nombre)
        except:
            try:
                creation(nombre)
                if not int(nombre[-1]):
                    print(nombre)
            except:
                pass


# Dataset creator


def bw(im):
    im_data = im.getdata()
    lst = []
    for pix in im_data:
        lst.append((pix[0] * 0.2125 + pix[1] * 0.7174 + pix[2] * 0.0721))
    new_image = Image.new("L", im.size)
    new_image.putdata(lst)
    return new_image, lst


def interpol(im, size):
    with im:
        px = im.load()
    listepix = {}
    width, height = im.size
    for x in range(width):
        if str(px[x, 0]) not in listepix:  # première longueur
            listepix[str(px[x, 0])] = 1
        else:
            listepix[str(px[x, 0])] += 1
        if str(px[x, height - 1]) not in listepix:  # deuxième longueur
            listepix[str(px[x, height - 1])] = 1
        else:
            listepix[str(px[x, height - 1])] += 1
    for y in range(height):
        if str(px[0, y]) not in listepix:  # première largeur
            listepix[str(px[0, y])] = 1
        else:
            listepix[str(px[0, y])] += 1
        if str(px[width - 1, y]) not in listepix:  # deuxième largeur
            listepix[str(px[width - 1, y])] = 1
        else:
            listepix[str(px[width - 1, y])] += 1

    background = max(listepix, key=listepix.get)[1:-1].split(',')  # Liste rgba de la couleur de fond
    background = [int(c) for c in background]

    new_image = Image.new("RGBA", (size, size), tuple(background))  # Crée le fond
    x = (size - width) // 2
    y = (size - height) // 2
    new_image.paste(im, (x, y), im)
    return new_image


def create_dataset(csv_path, pictures_path, dataset_path):
    chiffresvect = []
    labels = []
    # data = np.zeros([1, 900])
    with open(csv_path, newline='') as tableur:
        lines = csv.reader(tableur, delimiter=',', quotechar='|')
        for line in tqdm(lines):  # glob.glob("Captcha/*.png"):
            path = pictures_path.format(line[0][1:])
            image = Image.open(path)
            for i in range(0, (len(line) - 1) // 4):
                chiffre = int(line[0][i + 1])  # On récupère le chiffre à extraire
                labels.append(chiffre)
                imchiffre = image.copy().crop((int(line[i * 4 + 1]), int(line[i * 4 + 2]), int(line[i * 4 + 3]),
                                               int(line[i * 4 + 4])))  # On extrait le chiffre de l'image
                imchiffre = interpol(imchiffre, 19)
                imchiffre, liste = bw(imchiffre)  # Convertit l'image en noir et blanc
                # imchiffre.show()
                # noinspection PyTypeChecker
                imchiffre = np.array(imchiffre)
                # imchiffre = imchiffre.reshape((625,1))
                chiffresvect.append(
                    imchiffre)  # imchiffre pour avoir des images dans la base de données, liste pour avoir des listes
                # imchiffre.show()
                # print(chiffre)
            image.close()

    # Crée la base de données avec ce qu'il y a dans chiffresvect
    """
    try:
        f.close()
    except:
        pass
    """
    labelsall = [[0] * e + [1] + [0] * (9 - e) for e in labels]
    """
    labels0 = [int(e == 0) for e in labels]
    labels1 = [int(e == 1) for e in labels]
    labels2 = [int(e == 2) for e in labels]
    labels3 = [int(e == 3) for e in labels]
    labels4 = [int(e == 4) for e in labels]
    labels5 = [int(e == 5) for e in labels]
    labels6 = [int(e == 6) for e in labels]
    labels7 = [int(e == 7) for e in labels]
    labels8 = [int(e == 8) for e in labels]
    labels9 = [int(e == 9) for e in labels]
    """
    n = len(chiffresvect)
    frontiere = int(n * .8)
    with h5py.File(dataset_path, mode='w') as f:
        f.create_dataset("train", data=np.array(chiffresvect[:frontiere], dtype="f8"))
        f.create_dataset("test", data=np.array(chiffresvect[frontiere:], dtype="f8").T)
        f.create_dataset("trainlabel", data=np.array(labelsall[:frontiere], dtype="f8").T)
        f.create_dataset("testlabel", data=np.array(labelsall[frontiere:], dtype="f8"))
    """
    f.create_dataset("trainlabel0", data=np.array(labels0[:frontiere], dtype="f8"))
    f.create_dataset("testlabel0", data=np.array(labels0[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel1", data=np.array(labels1[:frontiere], dtype="f8"))
    f.create_dataset("testlabel1", data=np.array(labels1[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel2", data=np.array(labels2[:frontiere], dtype="f8"))
    f.create_dataset("testlabel2", data=np.array(labels2[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel3", data=np.array(labels3[:frontiere], dtype="f8"))
    f.create_dataset("testlabel3", data=np.array(labels3[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel4", data=np.array(labels4[:frontiere], dtype="f8"))
    f.create_dataset("testlabel4", data=np.array(labels4[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel5", data=np.array(labels5[:frontiere], dtype="f8"))
    f.create_dataset("testlabel5", data=np.array(labels5[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel6", data=np.array(labels6[:frontiere], dtype="f8"))
    f.create_dataset("testlabel6", data=np.array(labels6[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel7", data=np.array(labels7[:frontiere], dtype="f8"))
    f.create_dataset("testlabel7", data=np.array(labels7[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel8", data=np.array(labels8[:frontiere], dtype="f8"))
    f.create_dataset("testlabel8", data=np.array(labels8[frontiere:], dtype="f8"))
    f.create_dataset("trainlabel9", data=np.array(labels9[:frontiere], dtype="f8"))
    f.create_dataset("testlabel9", data=np.array(labels9[frontiere:], dtype="f8"))
    """
    # f.close()


# Entraînement

def initialisation(couches):
    C = len(couches)
    parametres = {}
    for c in range(1, C):
        parametres["W{}".format(c)] = np.random.randn(couches[c], couches[c - 1])
        parametres["b{}".format(c)] = np.random.randn(couches[c], 1)
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


def forward1(X, W, B):
    z = W[0].dot(X) + B[0]
    a = 1 / (1 + np.exp(-z))
    activations = [a]
    for i in range(1, len(W)):
        z = W[i].dot(a) + B[i]
        a = 1 / (1 + np.exp(-z))
        activations.append(a)
    return activations


def forward2(X, W, B):
    z = W[0].dot(X) + B[0]
    a = 1 / (1 + np.exp(-z))
    activations = [a]
    for i in range(1, len(W) - 1):
        z = W[i].dot(a) + B[i]
        a = 1 / (1 + np.exp(-z))
        activations.append(a)
    z = W[-1].dot(a) + B[-1]
    activations.append(softmax(z))
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
    return np.sum(-reponse_attendue * np.log(A + eps) - (1 - reponse_attendue) * np.log(1 - A + eps)) / len(
        reponse_attendue)


def neural_network(features, reponse_attendue, hidden_layers, learning_rate=0.1, n_iter=10000, features_test=None,
                   ytest=None, init=False, poids=None,
                   biais=None, nbiter=0, show=False):
    np.random.seed(0)
    couches = list(hidden_layers)
    couches = [features.shape[0]] + couches
    couches.append(reponse_attendue.shape[0])
    print(couches)
    if not init:
        parametres = initialisation(couches)
    else:
        parametres = {}
        for indice in range(len(poids)):
            parametres["W{}".format(indice + 1)] = poids[indice]
            parametres["b{}".format(indice + 1)] = biais[indice]

    train_loss, train_acc = [], []
    preclogloss, n_effectif_iter = 0, nbiter
    for _ in tqdm(range(1, n_iter + 1)):
        activations = forward_propagation(features, parametres)
        gradients = back_propagation(reponse_attendue, activations, parametres)
        parametres = update(gradients, parametres, 1 / _)  # 1/_ à la place de learning_rate
        n_effectif_iter += 1

        C = len(parametres) // 2
        logloss = log_loss(reponse_attendue, activations["A{}".format(C)])
        train_loss.append(logloss)
        # print(abs(logloss-preclogloss))
        y_pred = predict(features, parametres)
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
    plt.title("{} sous-couches\n{}".format(len(couches) - 2, str(couches[:-1]).replace(", ", "-")[1:-1]))
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train accuracy")
    plt.legend()
    plt.title("{} itérations".format(n_effectif_iter))
    plt.savefig("courbes/2_{}sc{}_{}".format(len(couches) - 1, str(couches[:-1]).replace(", ", "-")[1:-1],
                                                  datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
    if show:
        plt.show()
    return parametres, n_effectif_iter


def predict(features, parametres):
    C = len(parametres) // 2
    return forward_propagation(features, parametres)["A{}".format(C)] >= 0.5


def save(dataset, parametres, couches, chemin="modeles/"):
    n = len(couches) + 1
    poids = [parametres["W{}".format(_)] for _ in range(1, n)]
    biais = [parametres["b{}".format(_)] for _ in range(1, n)]
    temps = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    nom = "modele{}{}.txt".format(dataset, temps)
    t = open(chemin + nom, "a")
    t.write("W={}\n\n".format(poids))
    t.write("b={}".format(biais))
    t.close()


def normalise_x(vecteur):
    vecteur = vecteur.T.reshape(-1, vecteur.shape[0]) / vecteur.max()
    return vecteur


def normalise_y(vecteur):
    vecteur = vecteur.T.reshape(1, vecteur.shape[0])
    return vecteur


def pre_process_data(train_y, test_y):
    # Normalize
    # enc = OneHotEncoder(sparse=False, categories='auto')
    # train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    # test_y = enc.transform(test_y.reshape(len(test_y), -1))
    return train_y, test_y


def reprise_entrainement(nom):
    path = "modeles/{}.txt".format(nom)
    it = int(path.split("_")[3][:-2])
    layers = [int(nb_neurones) for nb_neurones in path.split("_")[1].split("-")]
    poids, biais = extract(path)

    # noinspection PyTypeChecker
    parametres, nb_total_iter = neural_network(X, Y["yall"], layers, n_iter=1000, learning_rate=0.01,
                                               features_test=Xtest,
                                               ytest=Y["ytestall"], init=True, poids=poids, biais=biais,
                                               nbiter=it)
    save(
        "2_{}".format(str(layers).replace(", ", "-")[1:-1]) + "_{}it_".format(nb_total_iter),
        parametres, layers)


def extract(path):
    file = open(path, "r")
    lignes = file.readlines()
    file.close()
    lignes_poids, lignes_biais = "", ""
    transition = False
    for ligne in lignes:
        if transition or "b=" in ligne:
            transition = True
            lignes_biais += ligne
        else:
            lignes_poids += ligne
    lignes_poids = lignes_poids[2:].replace("\n", "").replace("\t", "").replace(" ", "").replace("array", "np.array")
    lignes_biais = lignes_biais[2:].replace("\n", "").replace("\t", "").replace(" ", "").replace("array", "np.array")
    poids, biais = eval(lignes_poids), eval(lignes_biais)
    return poids, biais


# Test


def normalise(x, y, dy=False):
    x = x.T.reshape(x.shape[0] ** 2, 1) / x.max()
    if dy:
        y = y.T.reshape(1, y.shape[0])
    return x, y


def creer_test(taille=19, perturb=True):  # ou 25
    R, G, B = randint(20, 173), randint(20, 173), randint(20, 173)
    r_chiffre, g_chiffre, b_chiffre = randint(20, 173), randint(20, 173), randint(20, 173)
    im = Image.new("RGB", (taille, taille), (R, G, B))
    c = randint(0, 9)
    digit = choices(Liste_chiffres[c])[0]

    longueur, hauteur = digit.size
    x = (taille - longueur) // 2
    y = (taille - hauteur) // 2
    digit = change_couleur_chiffre(digit, r_chiffre, g_chiffre, b_chiffre)
    im.paste(digit, (x, y), digit)
    if perturb:
        x, y, xf, yf = randint(1, taille), randint(1, taille), randint(1, taille), randint(1, taille)
        draw = ImageDraw.Draw(im)
        epaisseur = randint(1, 3)
        r_ligne = randint(20, 173)  # Détermine la couleur de la ligne
        g_ligne = randint(20, 173)
        b_ligne = randint(20, 173)
        draw.line((x, y, xf, yf), fill=(r_ligne, g_ligne, b_ligne), width=epaisseur)  # Dessine la ligne
    # im.show()
    # im.save("test.png")
    im = bw(im)[0]
    im = np.array(im)
    return normalise(im, np.array(c))  # image, chiffre


def comparaison(N, perturb, paths, strat):
    """N : Nombre de chiffres à tester
    names : Les différents modèles
    """
    modeles = {}
    nb_modeles = len(paths)
    for indice in range(nb_modeles):
        modeles["result{}".format(indice)] = [0]
        modeles["W{}".format(indice)], modeles["b{}".format(indice)] = extract(
            "test_performances/" + paths[indice])
    for _ in range(N):
        im, c = creer_test(perturb=perturb)
        for j in range(nb_modeles):
            if strat[j] == 1:
                result = forward1(im, modeles["W{}".format(j)], modeles["b{}".format(j)])[-1]
                if (result >= .5) and c == j:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1] + 1)
                else:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1])
            elif strat[j] == 2:
                result = forward2(im, modeles["W{}".format(j)], modeles["b{}".format(j)])[-1]
                if c == list(result).index(max(result)):
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1] + 1)
                else:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1])
    return modeles


def test_pourcentage(paths, perturb=False, delete=False, nb_test=10000):
    nb_modeles = len(paths)
    scores = [0] * nb_modeles
    poids, biais = {}, {}
    for number in range(nb_modeles):
        poids["poids{}".format(number)], biais["biais{}".format(number)] = extract(
            "test_performances/" + paths[number])
    print("Testing \n{}".format("\n".join(paths)))
    for _ in range(nb_test):
        for number in range(nb_modeles):
            picture, c = creer_test(perturb=perturb)
            result = forward2(picture, poids["poids{}".format(number)], biais["biais{}".format(number)])[-1]
            if c == list(result).index(max(result)):
                scores[number] += 1
    if delete:
        for number in range(nb_modeles).__reversed__():
            if scores[number] != max(scores):
                print(paths[number])
                os.unlink("test_performances/" + paths[number])
    scores = ["{} : {} {}".format(i, e, paths[i]) for i, e in enumerate(np.array(scores) * 100 / nb_test)]
    print("\n".join(scores))


def test_courbe(paths, textes, perturb=False, nb_test=10000):
    """paths : liste : chemins des modèles
    textes : liste : légende des courbes"""
    nb_modeles = len(paths)
    modele = comparaison(nb_test, perturb, paths, [2] * nb_modeles)
    plt.close("all")
    for numero in range(nb_modeles):
        plt.plot(modele["result{}".format(numero)], label=textes[numero])
        # plt.plot(modele["result{}".format(i)], label = args[i].split("_")[3][:-2]+str("itérations") + " -- {}%".format(modele["result{}".format(i)][-1]*100/n))
        plt.title(
            "Scores cumulés " + ["sans perturbations", "avec perturbations"][perturb])
        # plt.axes(label = str(i))
    plt.legend()
    plt.show()
