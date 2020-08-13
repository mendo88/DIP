import numpy as np
import cv2 as cv
import math
from skimage.feature import hog
from matplotlib import pyplot as plt
from Utils import *

def enmascaradoHSV(original):
    #Trato de eliminar fondo mediante segmentacion HSV
    img = original.copy()

    #Ajuste de parametros
    #img_bin = defParametrosSegColor(img) #H=0-179 S=0-36 V=0-223

    imgHSV = cv.cvtColor(original, cv.COLOR_BGR2HSV)
    #Me quedo con el fondo en blanco
    lower = np.array([0, 0, 0], dtype = "uint8")
    upper = np.array([179, 36, 223], dtype = "uint8")
    img_bin = cv.inRange(imgHSV, lower, upper)

    #CIERRO para tomar todo el fondo
    c3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    c5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    c7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    c9 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    c11 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    img_bin7 = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, c9, iterations=10)

    #Invierto
    _,img_binInv = cv.threshold(img_bin7, 0, 255, cv.THRESH_BINARY_INV)
    
    #Control
    #hojas = cv.bitwise_and(original, original, mask=img_binInv)
    #cv.imshow("hojas", hojas)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return img_binInv