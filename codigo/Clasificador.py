import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
from Utils import *

def selector(mascTotal, mascSegmentada):
    #Saco la mascara del area enferma
    mascSupEnferma = mascTotal - mascSegmentada
    #cv.imshow("resta", mascSupEnferma)
    #Calculo areas
    supEnferma = cv.countNonZero(mascSupEnferma)
    supAnalizada = cv.countNonZero(mascTotal)

    porcEnfermo = (supEnferma*100)/supAnalizada
    if(porcEnfermo<0.1):
        #porcEnfermo = 0
        print("Planta SANA")
    else:
        print("Planta ENFERMA")

    print("Superficie analizada:", supAnalizada, "Superficie enferma:", supEnferma, "Porcentaje afectado:", porcEnfermo, "%")


def clasificar(imgEntrada, mascaraEntrada):
    imgHSV = cv.cvtColor(imgEntrada, cv.COLOR_BGR2HSV)
    #SEGMENTACION HSV PARA GENERAR MASCARA DE LA PARTE SANA
    #mascSegmentada = defParametrosSegColor(imgEntrada)          #H=35-103 S=10-255 V=0-255
    
    lower = np.array([35, 10, 0], dtype = "uint8")
    upper = np.array([103, 255, 255], dtype = "uint8")
    mascSegmentada = cv.inRange(imgHSV, lower, upper)

    selector(mascaraEntrada, mascSegmentada)