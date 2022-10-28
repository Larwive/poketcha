from PIL import Image, ImageDraw
import numpy as np
from random import randint, choices
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

chemin = "modeles/{}.txt".format("modele_400-400-400-10_0.958ac_6768it_9_03_06_2022__16_43_16")

# "modele2_1444-10_0.939ac_0it_5_28_05_2022__19_58_57"

chemin2 = "modeles/{}.txt".format("modele_400-400-400-10_0.964ac_10887it_9_03_06_2022__20_41_10")


# 'modele2_1024-1024-1024-10_0.931ac_0it_0_25_05_2022__15_01_21'

# 'modele2_512-512-512-10_0.911ac_0it_0_22_05_2022__21_20_29'


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


# W1, b1 = extract(path)
# W2, b2 = extract(path2)


def bw(picture):
    im_data = picture.getdata()
    lst = []
    for i in im_data:
        # lst.append(i[0]*0.299+i[1]*0.587+i[2]*0.114) ### Rec. 609-7 weights
        lst.append((i[0] * 0.2125 + i[1] * 0.7174 + i[2] * 0.0721))  # ## Rec. 709-6 weights
    new_image = Image.new("L", picture.size)
    new_image.putdata(lst)
    return new_image, lst


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


def normalise(x, y, dy=False):
    x = x.T.reshape(x.shape[0] ** 2, 1) / x.max()
    if dy:
        y = y.T.reshape(1, y.shape[0])
    return x, y


# Tests d'images
lettre = "D"
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
chiffre10 = Image.open("pixil/10")
chiffre11 = Image.open("pixil/11")
chiffre12 = Image.open("pixil/12")
chiffre13 = Image.open("pixil/13")
chiffre14 = Image.open("pixil/14")
chiffre15 = Image.open("pixil/15")
chiffre16 = Image.open("pixil/16")
chiffre17 = Image.open("pixil/17")
chiffre18 = Image.open("pixil/18")
chiffre19 = Image.open("pixil/19")
chiffre20 = Image.open("pixil/20")
chiffre21 = Image.open("pixil/21")
chiffre22 = Image.open("pixil/22")
chiffre23 = Image.open("pixil/23")
chiffre24 = Image.open("pixil/24")
chiffre25 = Image.open("pixil/25")
chiffre26 = Image.open("pixil/26")
chiffre27 = Image.open("pixil/27")
chiffre28 = Image.open("pixil/28")
chiffre29 = Image.open("pixil/29")

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
    (chiffre19, chiffre29),
)


def change_couleur_chiffre(picture, r_chiffre, g_chiffre, b_chiffre):
    """Change la couleur du chiffre à coller dans le captcha.
    """
    data = np.array(picture)
    red, green, blue, alpha = data.T
    fond = (red == 0) & (green == 0) & (blue == 0)
    data[..., :-1][fond.T] = (r_chiffre, g_chiffre, b_chiffre)
    return Image.fromarray(data)


# noinspection PyTypeChecker
def creer_test(taille=19, perturb=True):  # ou 25
    R, G, B = randint(20, 173), randint(20, 173), randint(20, 173)
    r_chiffre, g_chiffre, b_chiffre = randint(20, 173), randint(20, 173), randint(20, 173)
    picture = Image.new("RGB", (taille, taille), (R, G, B))
    c = randint(0, 9)
    digit = choices(Liste_chiffres[c])[0]

    longueur, hauteur = digit.size
    x = (taille - longueur) // 2
    y = (taille - hauteur) // 2
    digit = change_couleur_chiffre(digit, r_chiffre, g_chiffre, b_chiffre)
    picture.paste(digit, (x, y), digit)
    if perturb:
        x, y, xf, yf = randint(1, taille), randint(1, taille), randint(1, taille), randint(1, taille)
        draw = ImageDraw.Draw(im)
        epaisseur = randint(1, 3)
        r_ligne = randint(20, 173)  # Détermine la couleur de la ligne
        g_ligne = randint(20, 173)
        b_ligne = randint(20, 173)
        draw.line((x, y, xf, yf), fill=(r_ligne, g_ligne, b_ligne), width=epaisseur)  # Dessine la ligne
    # im.show()
    # im.save("D:/test.png")
    picture = bw(picture)[0]
    picture = np.array(picture)
    return normalise(picture, np.array(c))  # image, chiffre


def comparaison(N, perturb, paths, strat):
    """N : int : Nombre de chiffres à tester
    perturb : booléen : présence de perturbations
    paths : tuple : les différents modèles
    strat : tuple : liste des strats utilisées
    """
    modeles = {}
    nb_modeles = len(paths)
    for indice in range(nb_modeles):
        modeles["result{}".format(indice)] = [0]
        modeles["W{}".format(indice)], modeles["b{}".format(indice)] = extract(paths[indice])
    for _ in range(N):
        picture, c = creer_test(perturb=perturb)
        for j in range(nb_modeles):
            if strat[j] == 1:
                result = forward1(picture, modeles["W{}".format(j)], modeles["b{}".format(j)])[-1]
                if (result >= .5) and c == j:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1] + 1)
                else:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1])
            elif strat[j] == 2:
                result = forward2(picture, modeles["W{}".format(j)], modeles["b{}".format(j)])[-1]
                if c == list(result).index(max(result)):
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1] + 1)
                else:
                    modeles["result{}".format(j)].append(modeles["result{}".format(j)][-1])
    return modeles


# Stratégie 1
im, c = creer_test()
print(c)
resultat = forward1(im, W1, b1)[-1]
print(resultat)
print(list(resultat).index(max(resultat)))

# Stratégie 2
im, c = creer_test()
print(c)
resultat = forward2(im, W1, b1)[-1]
resultat2 = forward2(im, W2, b2)[-1]
print(resultat)
print(resultat2)
print("{} -- {}".format(list(resultat).index(max(resultat)), list(resultat2).index(max(resultat2))))

# Test pourcentage


def test_pourcentage(paths, perturb=False, nb_test=100000):
    nb_modeles = len(paths)
    scores = [0]*nb_modeles
    poids, biais = {}, {}
    for number in range(nb_modeles):
        poids["poids{}".format(number)], biais["biais{}".format(number)] = extract(path[i])
    for _ in range(nb_test):
        for number in range(nb_modeles):
            picture, c = creer_test(perturb=perturb)
            result = forward2(picture, poids["poids{}".format(number)], biais["biais{}".format(number)])[-1]
            if c == list(result).index(max(result)):
                scores[number] += 1
    scores = np.array(scores)*100/nb_test
    print(" -- ".join(scores))

# Test multi modèles


def test_courbe(paths, textes, perturb=False, nb_test=100000):
    """paths : liste : chemins des modèles
    textes : liste : légende des courbes"""
    nb_modeles = len(paths)
    modele = comparaison(nb_test, perturb, paths, [2]*nb_modeles)
    plt.close("all")
    for numero in range(nb_modeles):
        plt.plot(modele["result{}".format(numero)], label=textes[numero])
        # plt.plot(modele["result{}".format(i)], label = args[i].split("_")[3][:-2]+str("itérations") + " -- {}%".format(modele["result{}".format(i)][-1]*100/n))
        plt.title(
            "Scores cumulés " + ["sans perturbations", "avec perturbations"][perturb])
        # plt.axes(label = str(i))
    plt.legend()
    plt.show()


# Test de datasets
n = 0
score = 0
score2 = 0
with open("annotations.csv", newline='') as tableur:
    lines = csv.reader(tableur, delimiter=',', quotechar='|')
    for line in tqdm(lines):  # glob.glob("D:/Captcha/*.png"):
        im_path = "{}:/Creation/{}.png".format(lettre, line[0][1:])
        image = Image.open(im_path)
        for i in range(0, (len(line) - 1) // 4):
            n += 1
            chiffre = int(line[0][i + 1])  # On récupère le chiffre à extraire
            imchiffre = image.copy().crop((int(line[i * 4 + 1]), int(line[i * 4 + 2]), int(line[i * 4 + 3]),
                                           int(line[i * 4 + 4])))  # On extrait le chiffre de l'image
            imchiffre = interpol(imchiffre, 19)
            # imchiffre.show()
            imchiffre = bw(imchiffre)[0]  # Convertit l'image en noir et blanc
            # imchiffre.show()
            # noinspection PyTypeChecker
            imchiffre, _ = normalise(np.array(imchiffre), np.array(chiffre))
            imchiffre = np.array(imchiffre)
            resultat = forward2(imchiffre, W1, b1)[-1]
            if chiffre == list(resultat).index(max(resultat)):
                score += 1
            resultat = forward2(imchiffre, W2, b2)[-1]
            if chiffre == list(resultat).index(max(resultat)):
                score2 += 1
            # imchiffre = imchiffre.reshape((625,1))
            # imchiffre.show()
            # print(chiffre)
        image.close()

print("{} -- {}".format(100 * score / n, 100 * score2 / n))
