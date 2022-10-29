from functions import *
import os
import shutil
import sys
from random import randint

creer_nouveau_dataset = True
entrainement = True
test_performances = True
nb_captchas = 100
nb_iter = 100
pas_apprentissage: float = 0.01

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    if creer_nouveau_dataset:
        for filename in os.listdir("venv/creation"):
            file_path = os.path.join("venv/creation/", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.unlink(path_annotations)
        captcha_factory(4, nb_captchas // 3)
        captcha_factory(5, nb_captchas // 3)
        captcha_factory(6, nb_captchas // 3)
        create_dataset(path_annotations, path_pictures, path_dataset)

    if entrainement:
        with h5py.File(path_dataset, mode='r') as f:
            X = np.array(f["train"])
            Xtest = np.array(f["test"])
            X, Xtest = normalise_x(X), normalise_x(Xtest)
            Y = {"yall": (pre_process_data(np.array(f["trainlabel"]), np.array(f["testlabel"])))[0],
                 "ytestall": (pre_process_data(np.array(f["trainlabel"]), np.array(f["testlabel"])))[1]}

        for i in range(1, 11):
            couches = [0] * randint(3, 5)
            for j in range(len(couches)):
                couches[j] = randint(150, 350)
            # noinspection PyTypeChecker
            W_b, nombreffectifiter = neural_network(X, Y["yall"], couches, n_iter=nb_iter,
                                                    learning_rate=pas_apprentissage, features_test=Xtest,
                                                    ytest=Y["ytestall"], init=False, poids=None, biais=None)
            save("{}".format(str(couches).replace(", ", "-")[1:-1]) + "_{}ac_{}it_".format(round(accuracy, 3),
                                                                                           nombreffectifiter), W_b,
                 couches)
    if test_performances:
        chemins = os.listdir("venv/test_performances")
        test_pourcentage(chemins, True)
        # texte = [""]*50
        # test_courbe(chemins,
        # texte, True)
