from PIL import Image
import numpy as np
import csv
import h5py
# import glob
from tqdm import tqdm


def bw(im):
    im_data = im.getdata()
    lst = []
    for pix in im_data:
        # lst.append(i[0]*0.299+i[1]*0.587+i[2]*0.114) ### Rec. 609-7 weights
        lst.append((pix[0] * 0.2125 + pix[1] * 0.7174 + pix[2] * 0.0721))  # ## Rec. 709-6 weights
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


# Crée un tableau contenant les chiffres extraits

def create_dataset(csv_path, pictures_path, dataset_path):
    chiffresvect = []
    labels = []
    # data = np.zeros([1, 900])
    with open(csv_path, newline='') as tableur:
        lines = csv.reader(tableur, delimiter=',', quotechar='|')
        for line in tqdm(lines):  # glob.glob("D:/Captcha/*.png"):
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
    try:
        f.close()
    except:
        pass

    f = h5py.File(dataset_path, mode='w')

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
    f.close()
##
# f = h5py.File(dataset_path, mode='r')
