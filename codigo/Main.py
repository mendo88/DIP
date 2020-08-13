import numpy as np
import cv2 as cv
import math
from skimage.feature import hog
from matplotlib import pyplot as plt
from Utils import *
from EnmascaradoGrises import *
from EnmascaradoColor import *
from Clasificador import *




if __name__ == "__main__":
    #Flag si elimino suelo o no / flag if delete floor
    delSuelo = True
    #Path de la imagen
    path = "../train/segmentacion/C1P1E2"
    extension = ".jpg"

    #Cargo la imagen en RGB / Load image in RBG
    original = cv.imread(path+extension)

    #Reescalo al 30% / rescale to 30%
    img = cv.resize(original, (-1,-1), None, 0.3, 0.3)

    #Genero una copia en escala de grises / Make a copy to gray scale
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Pre procesamiento, eliminacion de parte del fondo
    #Mascara sin el piso, tambien se utiliza para optimizar imagen final en caso de que se generen
    #huecos espurios dentro de la hoja principal
    mascara1 = enmascaradoHSV(img)

    if (delSuelo):
        #Pre procesamiento - Eliminacion del piso
        imgPrim = cv.bitwise_and(img, img, mask=mascara1)
    else:
        #Sin eliminar piso
        imgPrim=img.copy()

################################################################################################################

    #Segundo procesamiento, eliminacion de algunas hojas y fondo
    #Segun el caso se enmascara la imagen con la primer mascara o no y esta es la entrada de este proceso
    #Segunda mascara

    # Leaf removal and the background
    mascara = enmascaradoEscalaGrises(imgPrim, imgGray)
    enmasc = cv.bitwise_and(imgPrim, imgPrim, mask=mascara)

################################################################################################################

    #Tercer procesamiento, separacion de hojas y generacion de contornos
    # Leaf separation and contour generation

    contours = enmascaradoPiola(enmasc)
    #Dibujo contornos sobre la primer mascara
    mascara = cv.drawContours(mascara, contours, -1, (0,0,0),2)

    #Busco el contorno de mayor area / Check for the contour with the higher area
    #Saco la máscara de la hoja principal (area) y su centro
    # Get the main leaf mask and his center
    componenteMaximaArea, centroMax, boundingData = maxConvex(mascara)

    #Enmascaro la imagen original quedandome con la hoja principal / Apply the mask
    result = cv.bitwise_and(img, img, mask=componenteMaximaArea)
    
    #TEST PARA SACAR MASCARA OPTIMA
    optim = cv.bitwise_and(componenteMaximaArea,mascara1)
    #A maxima le resto optima para obtener los huecos
    optim2 = componenteMaximaArea - optim
    mascaraFinal = cv.bitwise_or(optim, optim2)
    #METER EL RETOQUE ACA
    mascaraFinal = retoqueCierre(mascaraFinal)
    imgFinal = cv.bitwise_and(img, img, mask=mascaraFinal)
    
    clasificar(imgFinal, mascaraFinal)
    
#
    #cv.imshow("original", img)
    #cv.imshow("componenteMaximaArea", componenteMaximaArea)
    #cv.imshow("result", result)
    #cv.imshow("mascara1", mascara1)
    #cv.imshow("optima", optim)
    #cv.imshow("optima2", optim2)
    #cv.imshow("mascarafinal",mascaraFinal)
    cv.imshow("imgFinal", imgFinal)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("image.png", imgFinal)

    #####   PRUEBA DESEMPEÑO  ###### Performance test with Intersection over Union
    #PRUEBA CON GT / TEST WITH GROUND TRUTH
    #pathGT = "_GT.png"
    #imgGT = cv.imread(path+pathGT)
    #imgGT = cv.resize(imgGT, (-1,-1), None, 0.3, 0.3)
    #IoU(imgGT, mascaraFinal)
    ################################

    
    
    