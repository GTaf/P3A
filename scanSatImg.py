from osgeo import gdal
import numpy as np
import os

dataset = gdal.Open('../../Corse/Orthorectifiee/CORSE_ORTHO.tif', gdal.GA_ReadOnly)

print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                             dataset.GetDriver().LongName))
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))


if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

xImgSize = dataset.RasterXSize
yImgSize = dataset.RasterYSize

xActiveWindowSize = 1000
yActiveWindowSize = 1000
xSuperpositionSize = 100
ySuperpositionSize = 100
xbins = xImgSize//(xActiveWindowSize-xSuperpositionSize)
if xImgSize % (xActiveWindowSize-xSuperpositionSize) != 0:
    xbins += 1
ybins = yImgSize//(yActiveWindowSize-ySuperpositionSize)
if yImgSize % (yActiveWindowSize-ySuperpositionSize) != 0:
    ybins += 1
import matplotlib.pyplot as plt
for i in range(0,xImgSize,xActiveWindowSize-xSuperpositionSize):
    for j in range(0, yImgSize, yActiveWindowSize - ySuperpositionSize):
        xsize = xActiveWindowSize
        if xActiveWindowSize > xImgSize - i:
            xsize = xImgSize - i
        ysize=yActiveWindowSize
        if yActiveWindowSize > yImgSize - j:
            ysize = yImgSize - j
        window = band.ReadAsArray(i, j, xsize, ysize)
        print("Reading image at offset = ({}, {})".format(i, j))
        #plt.imshow(window)
        #plt.show()
