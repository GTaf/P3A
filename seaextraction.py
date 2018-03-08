from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def normalize(image,max):
    if np.max(image) == np.min(image):
        return image*0.
    return max*1./(np.max(image)-np.min(image))*(image-np.min(image))

#La fonction renvoie un couple (booleen,tableau). Si la mer est presente, ce qui est renvoye est True et le masque, sinon, on renvoie False et 0
def getSeaMask(dataset):
    ###########################Temps 1: sous-echantillonnage de l'image###########################################
    #Le but est d'obtenir une image de taille raisonnable, sur laquelle effectuer des calculs plus rapides.

    #Calculs preliminaires pour le parcours de l'image
    band = dataset.GetRasterBand(1)
    xImgSize = dataset.RasterXSize
    yImgSize = dataset.RasterYSize
    xActiveWindowSize = 100
    yActiveWindowSize = 100
    xSuperpositionSize = 0  # 100
    ySuperpositionSize = 0  # 100
    xbins = xImgSize // (xActiveWindowSize - xSuperpositionSize)
    if xImgSize % (xActiveWindowSize - xSuperpositionSize) != 0:
        xbins += 1
    ybins = yImgSize // (yActiveWindowSize - ySuperpositionSize)
    if yImgSize % (yActiveWindowSize - ySuperpositionSize) != 0:
        ybins += 1
    xRange = xImgSize / (xActiveWindowSize - xSuperpositionSize)
    yRange = yImgSize / (yActiveWindowSize - ySuperpositionSize)

    #Generation de l'image sous-echantillonee
    map = np.zeros((yRange + 1, xRange + 1))
    for i in range(0, xImgSize, xActiveWindowSize - xSuperpositionSize):
        print('i')
        for j in range(0, yImgSize, yActiveWindowSize - ySuperpositionSize):
            xsize = xActiveWindowSize
            if xActiveWindowSize > xImgSize - i:
                xsize = xImgSize - i
            ysize = yActiveWindowSize
            if yActiveWindowSize > yImgSize - j:
                ysize = yImgSize - j
            window = band.ReadAsArray(i, j, xsize, ysize)
            map[j / (yActiveWindowSize - ySuperpositionSize), i / (xActiveWindowSize - xSuperpositionSize)] = np.sum(window)
    print('done')
    ###########################Temps 2: evaluation de l'histogramme d'intensite, calcul de la limite d'intensite mer-terre###########################################
    bins, edges = np.histogram(normalize(map.copy(), 255), bins=255)

    # Lissage de l'histogramme par un filtre median
    lisse = medfilt(bins, 3)

    # On cherche s'il y a de l'eau : s'il y a de l'eau, la distribution d'intensites lumineuse comporte deux pics au lieu d'un.
    #Pour trouver ou est le pic, on cherche a partitionner l'histogramme en deux. On evalue le gain d'information en fonction de la position de la coupure.
    #Raisonner sur le gain d'information permet de lisser la courbe et d'obtenir des resultats plus propres qu'avec une recherche de maxima et de minima sur les intensites lumineuses elles-memes
    var = np.var(lisse)
    informationgain=[]
    for i in range(len(lisse)):
        varg = np.var(lisse[:i])
        vard = np.var(lisse[i:])
        informationgain=informationgain + [var - len(lisse[:i])*1./len(lisse)*varg - len(lisse[i:])*1./len(lisse)*vard]

    informationgain = medfilt(informationgain, 3)
    #On recherche les minima locaux. S'il n'y en n'a qu'un, la distribution d'intensite est centree sur une valeur unique et il n'y a pas d'eau.
    #S'il y en a deux, il y a de l'eau. On recherche alors, entre ces deux minima, la valeur minimale dans l'histogramme des intensites, c'est celle qui sera la limite entre eau et terre.
    #Recherche des minima
    minipos=[]
    minivalue=[]
    for i in range(1,len(informationgain)-1):
        if informationgain[i] <= informationgain[i-1] and informationgain[i] < informationgain[i+1]:
            minivalue = minivalue + [informationgain[i]]
            minipos = minipos + [i]
    print(len(minivalue))
    if len(minivalue) != 2:           #Soit len(minivalue)==1 : il n'y a que de la terre, soit l'algorithme a rencontre une situation particuliere et ne filtre pas.
        #Decommenter pour inspecter les histogrammes
        """
        plt.plot(edges[:-1], informationgain)
        plt.title("Evolution du gain d'information en fonction de l'intensite de coupure")
        plt.xlabel("Intensite lumineuse")
        plt.ylabel("Gain")
        plt.show()
        plt.plot(edges[:-1], lisse)
        plt.title("Histogramme d'intensite lumineuse")
        plt.xlabel("Intensite lumineuse")
        plt.ylabel("Nb d'occurences")
        plt.show()
        """
        return False, 0
    else:                           #La mer a ete detectee, on renvoie un masque binaire : 0 pour la mer, 255 pour la terre
        #On determine la valeur seuil
        threshold = np.argmin(bins[minipos[0]:minipos[1]])+minipos[0]
        print(minipos[0])
        print(minipos[1])
        print(threshold)
        map = normalize(map,255)
        seamap = np.zeros(map.shape)
        seamap[map < threshold] = 0
        seamap[map >= threshold] = 255

        #Decommenter pour inspecter les histogrammes
        """
        plt.scatter([minipos[0], minipos[1]], [informationgain[minipos[0]], informationgain[minipos[1]]], c='r')
        plt.plot(edges[:-1], informationgain)
        plt.title("Evolution du gain d'information en fonction de l'intensite de coupure")
        plt.xlabel("Intensite lumineuse")
        plt.ylabel("Gain")
        plt.show()
        plt.plot(edges[:-1], lisse)
        plt.scatter([threshold], [lisse[threshold]], c='r')
        plt.title("Histogramme d'intensite lumineuse")
        plt.xlabel("Intensite lumineuse")
        plt.ylabel("Nb d'occurences")
        plt.show()
        """
        return True, seamap
