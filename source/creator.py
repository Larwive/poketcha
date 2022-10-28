from random import randint, choices
from PIL import Image, ImageDraw
import operator
import numpy as np
import csv


path = "../creation"


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
# Met les images dans une liste
nombre_chaque = [0 for _ in range(999999)]


def creation(number, show=False):
    number = str(number)
    r = randint(20, 173)
    g = randint(20, 173)
    b = randint(20, 173)
    a = choices([200, 255], weights=[2355, 1])[0]
    nombre_de_chiffres = len(number)  # Détermine le nombre de chiffres du captcha entre 4 et 6 selon la place disponible
    w = randint(max(118, 10 + 20 * nombre_de_chiffres), 189)
    h = randint(40, 59)

    new_captcha = Image.new("RGBA", (w, h), (r, g, b, a))  # Crée le fond
    # new_captcha.show() #Affiche la base du captcha (fond)

    ligne_csv = []
    r_chiffre, g_chiffre, b_chiffre = r, g, b
    while (r_chiffre, g_chiffre, b_chiffre) == (r, g, b):  # Permet de s'assurer que les chiffres n'aient pas la même couleur que le fond
        r_chiffre = randint(20, 173)
        g_chiffre = randint(20, 173)
        b_chiffre = randint(20, 173)

    espace_en_plus = int((w - 2 * 5 - 20 * nombre_de_chiffres) / (
            2 * nombre_de_chiffres - 2))  # Détermine la marge en plus en abscisse qu'un chiffre peut prendre

    chiffre = choices(Liste_chiffres[int(number[0])])[0]  # Choisit une police aléatoire
    chiffre = change_couleur_chiffre(chiffre, r_chiffre, g_chiffre, b_chiffre)  # Change la couleur du chiffre
    longueur, hauteur = chiffre.size
    longueur_precedente = longueur

    x = randint(5, 25 - longueur)
    y = randint(5, h - 5 - hauteur)
    new_captcha.paste(chiffre, (x, y), chiffre)
    appends(ligne_csv, x - 2, y - 2, x + longueur + 2, y + hauteur + 2)
    for i in range(1, nombre_de_chiffres - 1):
        chiffre = choices(Liste_chiffres[int(number[i])])[0]  # Choisit une police aléatoire
        chiffre = change_couleur_chiffre(chiffre, r_chiffre, g_chiffre, b_chiffre)  # Change la couleur du chiffre
        longueur, hauteur = chiffre.size
        x = randint(x + longueur_precedente + 2, 5 + 20 * i + espace_en_plus * 2 * (i + 1))
        y = randint(5, h - 5 - hauteur)
        appends(ligne_csv, x - 2, y - 2, x + longueur + 2, y + hauteur + 2)
        longueur_precedente = longueur
        new_captcha.paste(chiffre, (x, y), chiffre)

    chiffre = choices(Liste_chiffres[int(number[-1])])[0]  # Choisit un chiffre aléatoire
    chiffre = change_couleur_chiffre(chiffre, r_chiffre, g_chiffre, b_chiffre)  # Change la couleur du chiffre
    longueur, hauteur = chiffre.size
    x = randint(x + longueur_precedente + 2, w - 5 - longueur_precedente)
    y = randint(5, h - 5 - hauteur)
    appends(ligne_csv, x - 2, y - 2, x + longueur + 2, y + hauteur + 2)

    new_captcha.paste(chiffre, (x, y), chiffre)

    nb_lignes = randint(4, 5) + 2
    draw = ImageDraw.Draw(new_captcha)

    for i in range(nb_lignes):
        x, y, xp, yp = randint(0, w), randint(0, h), randint(0, w), randint(0,
                                                                            h)  # On détermine les coordonnées des extrémités du segment
        epaisseur = randint(0, i // 2)  # Détermine l'épaisseur de la ligne
        r_ligne = randint(20, 173)  # Détermine la couleur de la ligne
        g_ligne = randint(20, 173)
        b_ligne = randint(20, 173)
        draw.line((x, y, xp, yp), fill=(r_ligne, g_ligne, b_ligne), width=epaisseur)  # Dessine la ligne

    x, y = randint(0, w - 50), randint(0, h - 20)
    xp, yp = randint(x, w), randint(y, h)
    angle_debut = randint(0, 89)
    angle_fin = 180 - angle_debut
    epaisseur = randint(1, 3)
    r_arc = randint(20, 173)  # Détermine la couleur de l'arc
    g_arc = randint(20, 173)
    b_arc = randint(20, 173)
    draw.arc((x, y, xp, yp), angle_debut, angle_fin, fill=(r_arc, g_arc, b_arc), width=epaisseur)  # Dessine l'arc

    new_captcha.save("creation/{}#{}.png".format(number, nombre_chaque[int(number)]))
    if show:
        new_captcha = Image.open("creation/{}#{}.png".format(number, nombre_chaque[int(number)]))
        new_captcha.show()

    with open("annotations.csv", 'a', encoding='UTF8', newline='') as f:
        '''
        lines = csv.reader(f)
        for line in lines:
            #line = line.rstrip("\n")
            #print(line)
            if "#{}".format(number) in line[0]:
                print("OHOH")
        '''
        writer = csv.writer(f)
        writer.writerow(["#{}#{}".format(number, nombre_chaque[int(number)])] + ligne_csv)
        f.close()
    nombre_chaque[int(number)] += 1


# noinspection PyBroadException
def captcha_factory(nb_chiffres, quantite):
    for _ in range(quantite):
        nombre = randint(0, int(nb_chiffres*"9"))
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
