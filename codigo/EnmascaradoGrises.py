import numpy as np
import cv2 as cv
import math
from skimage.feature import hog
from matplotlib import pyplot as plt
from Utils import *


def enmascaradoEscalaGrises(img, imgGray):
    #Extraigo bordes y los ensancho con morfologia, tapo huecos tb

    imgBordes = bordes_laplaciano(imgGray, umbral1=107, umbral2=255, kernel=3)

    circulo5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    imgDilate = cv.dilate(imgBordes,circulo5, iterations=1)
    imgDilate = cv.morphologyEx(imgBordes, cv.MORPH_CLOSE, circulo5, iterations=2)

    #Invierto el resultado para generar la mascara y cierro huecos
    imgMask = cv.bitwise_not(imgDilate)
    circulo3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    imgMask = cv.dilate(imgMask, circulo3, iterations=3)
    imgMask = cv.morphologyEx(imgMask, cv.MORPH_ERODE, circulo3, iterations=2)

    #Tomo la componente conexa de mayor tamaño y genero una nueva mascara a partir de ahi
    labels,_,_ = maxConvex(imgMask)
    labels2 = cv.morphologyEx(labels, cv.MORPH_ERODE, circulo5, iterations=3)
    #Repito proceso
    labels2,_,_ = maxConvex(labels2)
    mascara = cv.morphologyEx(labels2, cv.MORPH_CLOSE, circulo5, iterations=8)
    mascara = cv.morphologyEx(mascara, cv.MORPH_DILATE, circulo5, iterations=3)
    
    return mascara


def enmascaradoPiola(img):
    #Etapa uno, suavizar fondo e interior de las hojas preservado bordes
    imgBlur = cv.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
    
    #Deformo los colores preservando solo los bordes, dando un efecto de acuarela, lo q realza el borde
    imgAc = cv.stylization(imgBlur, sigma_s=200, sigma_r=0.7)

    #Paso a escala de grises
    imgGray = cv.cvtColor(imgAc, cv.COLOR_BGR2GRAY)

    #Filtrado para suavizar mas el fondo
    imgG = cv.GaussianBlur(imgGray, (5,5), 0.5)

    #Expansion de bordes por dilatacion
    ee = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    imgDil = cv.morphologyEx(imgG, cv.MORPH_DILATE, ee, iterations=1)

    #Hago un filtrado de promedio para eliminar bordes internos de la hoja debido a la textura
    img1Fil = cv.boxFilter(imgDil, -1, (5,5))

    #Binarizo con thresh adaptativo
    img1Bin = cv.adaptiveThreshold(img1Fil, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)


    #Busqueda y dibujo de contornos
    imgContornos = img.copy()
    contours, _ = cv.findContours(img1Bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  #Contornos externos
    cv.drawContours(imgContornos, contours, -1, (0,0,255),2)


    return contours