"""
Created on Mon Nov 20 17:25:14 2017

@author: gtaf
"""

import numpy as np
import os
import pandas as pd
from osgeo import gdal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

batch_size = 5
num_classes = 2
epochs = 5

img_rows, img_cols = 81, 81

pictures = []
labels = []
filenames=[]
#Loading the training vehicules
for element in os.listdir('../Base_Defi/Automobile/Valide/'):
    filenames.append(element[:-4])
    if not (element.endswith(".tif") or element.endswith(".tiff")):
        continue
    filenames.append(element[:-4])
    dataset = gdal.Open('../Base_Defi/Automobile/Valide/' + element, gdal.GA_ReadOnly)
    channel = np.array(dataset.GetRasterBand(1).ReadAsArray())
    if channel.shape == (81, 81):
        pictures.append(channel)
        labels.append(0)

nbVehicules = len(pictures)
print("vehicules", nbVehicules)
#Loading the training background

for element in os.listdir('../Base_Defi/Fond/'):
    if not (element.endswith(".tif") or element.endswith(".tiff")):
        continue
    dataset = gdal.Open('../Base_Defi/Fond/' + element, gdal.GA_ReadOnly)
    channel = np.array(dataset.GetRasterBand(1).ReadAsArray())
    if channel.shape == (81, 81):
        pictures.append(channel)
        pictures.append(np.flip(channel, 1))
        pictures.append(np.flip(channel, 0))
        pictures.append(np.flip(np.flip(channel, 0), 1))
        labels.append(1)
        labels.append(1)
        labels.append(1)
        labels.append(1)

nbFonds = len(pictures) - nbVehicules
print("fonds", nbFonds)

#Change image into its descriptor
from skimage.feature import hog

hog_descriptor = []
from sklearn.decomposition import PCA
pcaDesc = []
pictures = np.array(pictures)
for image in pictures:
    hog_descriptor.append(hog(image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3), transform_sqrt=True))
    pca = PCA(n_components=7)
    pca.fit(image)
    coke_gray_pca = pca.fit_transform(image)
    pcaDesc.append(coke_gray_pca.flatten())

from sklearn import svm
my_svm = svm.SVC()
my_svm.fit(pcaDesc, labels)
from sklearn.externals import joblib
print('saving svm...')
joblib.dump(my_svm, 'firstsvmpca' + '.pkl')
print('done')
wish = 'none'
while wish != 'stop':
    wish = input('select picture to print')
    if wish == 'stop':
        break
    (H, hogImage) = hog(pictures[int(wish)], orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3),
                        transform_sqrt=True,
                        visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(pictures[int(wish)], cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')
    from skimage import data, color, exposure

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    x=input('x')
    y=input('y')
    plt.imshow(pictures[int(wish)][y:y+15,x:x+15], cmap=plt.cm.gray)
    plt.show()
    (H, hogImage) = hog(pictures[int(wish)][y:y+15,x:x+15], orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3),
    transform_sqrt = True,
    visualise = True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(pictures[int(wish)][y:y+15,x:x+15], cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')
    from skimage import data, color, exposure

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    from PIL import *
    from scipy import misc

    misc.imsave('outfile2.jpg', pictures[int(wish)].astype(np.uint16))
    misc.imsave('littleimages/'+filenames[wish]+'.jpg', pictures[int(wish)][y:y+15,x:x+15].astype(np.uint16))
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    pca = PCA(n_components=2)
    X_r = pca.fit(pictures[int(wish)][y:y+15,x:x+15]).transform(pictures[int(wish)][y:y+15,x:x+15])


    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    a=X_r[:, 0]
    b=X_r[:, 1]
    print(a)
    plt.scatter(a,b)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')


    plt.show()

    from sklearn.decomposition import PCA
    from pylab import *
    from skimage import data, io, color

    link = "http://perierga.gr/wp-content/uploads/2012/01/coca_cola.jpg"
    coke_gray = pictures[int(wish)][y:y+15,x:x+15]

    subplot(2, 2, 1)
    io.imshow(coke_gray, cmap=plt.cm.gray)
    xlabel('Original Image')
    for i in range(15):
        n_comp = i
        pca = PCA(n_components=n_comp)
        pca.fit(coke_gray)
        coke_gray_pca = pca.fit_transform(coke_gray)
        plt.subplot(2,20,i+1)
        plt.imshow(coke_gray_pca, cmap=plt.cm.gray)

        coke_gray_restored = pca.inverse_transform(coke_gray_pca)
        #subplot(2, 2, i+1)
        plt.subplot(2, 20, i+21)
        plt.imshow(coke_gray_restored, cmap=plt.cm.gray)

        xlabel('Restored image n_components = %s' % n_comp)
        print 'Variance retained %s %%' % (
        (1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_)) * 100)
        print 'Compression Ratio %s %%' % (float(size(coke_gray_pca)) / size(coke_gray) * 100)
    show()

from sklearn import svm
my_svm = svm.SVC()
my_svm.fit(hog_descriptor, labels)
from sklearn.externals import joblib
print('saving svm...')
joblib.dump(my_svm, 'firstsvm' + '.pkl')
print('done')
