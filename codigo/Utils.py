import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure



def param_trackBar(trackbarName, windowName, valor_inicial=0, delta=1, negativo=False):
    pos = cv.getTrackbarPos(trackbarName, windowName)
    if(negativo):
        pos = cv.getTrackbarPos(trackbarName, windowName) - 50
    val = valor_inicial + (pos * delta)
    return val

# cv.createTrackBar se le debe pasar una funcion como parametro, como
# se necesita solo la posicion del track bar, le pasamos una que no
# hace nada
def track_change(val):
    pass

def param_trackBar(trackbarName, windowName, valor_inicial=0, delta=1, negativo=False):
    pos = cv.getTrackbarPos(trackbarName, windowName)
    if(negativo):
        pos = cv.getTrackbarPos(trackbarName, windowName) - 50
    val = valor_inicial + (pos * delta)
    return val


def determinarHOG(image, graficar=False):

    fd, hog_image = hog(image, orientations=10, pixels_per_cell=(7,7), cells_per_block=(1,1), visualize=True, multichannel=True, transform_sqrt=True, block_norm='L1')

    if(graficar):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return fd, hog_image

def bordes_LoG(img, umbral1, umbral2):
    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])
    output_image = cv.filter2D(img, -1, kernel)
    _, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image

def bordes_laplaciano(img, tipoDato=-1, kernel=3, threshold=True, umbral1=0, umbral2=255):
    """No ensancha los bordes"""
    output_image = cv.Laplacian(img, cv.CV_8U, ksize=kernel)
    if(threshold):
        _, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def defParametrosHOG(img):
    metodo_normalizacion = ['L1', 'L1-sqrt', 'L2', 'L2-Hys']
    visualizacion = [True, False]
    transformacion_sqrt = [True, False]

    cv.namedWindow("control")
    cv.createTrackbar("nro_orientaciones", "control", 1, 100, track_change)
    cv.createTrackbar("pixels_por_cel", "control", 1, 100, track_change)
    cv.setTrackbarPos("pixels_por_cel", "control", 16)
    cv.createTrackbar("cel_por_block", "control", 1, 100, track_change)
    cv.setTrackbarPos("cel_por_block", "control", 1)
    cv.createTrackbar("norm_method", "control", 0, 3, track_change)
    cv.createTrackbar("visualize", "control", 0, 1, track_change)
    cv.createTrackbar("transform_sqrt", "control", 0, 1, track_change)

    while(True):
        nro_o = param_trackBar("nro_orientaciones", "control")

        pixels_por_cel = param_trackBar("pixels_por_cel", "control")
        px_p_c = (pixels_por_cel, pixels_por_cel)

        cel_por_block = param_trackBar("cel_por_block", "control")
        cel_p_b = (cel_por_block, cel_por_block)

        nm = param_trackBar("norm_method", "control")
        norm_method = metodo_normalizacion[nm]

        v = param_trackBar("visualize", "control")
        visualize = visualizacion[v]

        tsqrt = param_trackBar("transform_sqrt", "control")
        trans_sqrt = transformacion_sqrt[tsqrt]

        _, img_hog = hog(img, orientations=nro_o, pixels_per_cell=px_p_c, cells_per_block=cel_p_b, visualize=visualize, multichannel=True, transform_sqrt=trans_sqrt, block_norm=norm_method)
        
        img_hog = cv.resize(img_hog, (-1,-1), None, 0.3, 0.3) 
        cv.imshow("img_hog", img_hog)
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()

def defParametrosCanny(original, gray=True):
    if(gray == True):
        img = original.copy()
    else:
        img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    
    L2Gradient = [False, True]
    apertura = [3, 5, 7]
    cv.namedWindow("control")
    cv.createTrackbar("umbralCanny1", "control", 0, 10000, track_change)
    cv.createTrackbar("umbralCanny2", "control", 0, 10000, track_change)
    cv.createTrackbar("apSize", "control", 0, 2, track_change)
    cv.createTrackbar("L2", "control", 0, 1, track_change)

    while(True):
        umbral1 = param_trackBar("umbralCanny1", "control")
        umbral2 = param_trackBar("umbralCanny2", "control")
        ap = param_trackBar("apSize", "control")
        apSize = apertura[ap]
        L2 = param_trackBar("L2", "control")
        L2Grad = L2Gradient[L2]
        img_canny = cv.Canny(img, umbral1, umbral2, apertureSize=apSize, L2gradient=L2Grad)
        
        cv.imshow("canny", img_canny)
        
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()
    return img_canny

def defParametrosMorfologia(img):
    ee = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    img_morph = cv.dilate(img, ee, iterations=2)

    return img_morph

def defParametrosLaplaciano(img, aGris=False):
    if aGris:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow("control")
    cv.createTrackbar("umbral1", "control", 0, 255, track_change)
    cv.createTrackbar("umbral2", "control", 0, 255, track_change)
    cv.createTrackbar("kernel", "control", 3, 29, track_change)

    while(True):
        umbral1 = param_trackBar("umbral1", "control")
        umbral2 = param_trackBar("umbral1", "control")
        kernel = param_trackBar("kernel", "control")
        output_image = bordes_laplaciano(img, umbral1=umbral1, umbral2=umbral2, kernel=kernel)
       
        cv.imshow("Laplaciano", output_image)
        
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()


def defParametrosLoG(img):
    cv.namedWindow("control")
    cv.createTrackbar("umbral1", "control", 0, 255, track_change)
    cv.createTrackbar("umbral2", "control", 0, 255, track_change)
    while(True):
        umbral1 = param_trackBar("umbral1", "control")
        umbral2 = param_trackBar("umbral1", "control")

        output = bordes_LoG(img, umbral1, umbral2)
        cv.imshow("LoG", output)
        
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()

def defParametrosThreshold(img, gray=True):
    if(gray):
        img_gray = img.copy()
    else:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.namedWindow("controlB")
    cv.createTrackbar("umbral1", "controlB", 0, 255, track_change)
    cv.createTrackbar("umbral2", "controlB", 0, 255, track_change)

    while(True):
        umbral1 = param_trackBar("umbral1", "controlB")
        umbral2 = param_trackBar("umbral2", "controlB")
        
        _,img_bin = cv.threshold(img_gray, umbral1, umbral2, cv.THRESH_BINARY)
        
        cv.imshow("bin", img_bin)
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()

def defParametroAdapTresh(img):
    cv.namedWindow("controlB")
    cv.createTrackbar("umbral1", "controlB", 0, 255, track_change)
    cv.createTrackbar("umbral2", "controlB", 0, 255, track_change)

    while(True):
        umbral1 = param_trackBar("umbral1", "controlB")
        umbral2 = param_trackBar("umbral2", "controlB")
        
        _,img_bin = cv.adaptiveThreshold(img, umbral1, umbral2)
        
        cv.imshow("bin", img_bin)
        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()


def defParametrosEdgePreservingFilter(img):
    cv.namedWindow("control")
    cv.createTrackbar("sigma_s", "control", 0, 200, track_change)
    cv.createTrackbar("sigma_r", "control", 0, 100, track_change)
    while(True):
        sigma_s = param_trackBar("sigma_s", "control")
        sigma_r = param_trackBar("sigma_r", "control", delta=0.01)

        output = cv.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

        cv.imshow("resultado", output)
        cv.imshow("original", img)
        key = cv.waitKey(1) & 0XFF
        if(key == ord('q')):
            break
    cv.destroyAllWindows()

def defParametrosStylization(img):
    cv.namedWindow("control")
    cv.createTrackbar("sigma_s", "control", 0, 200, track_change)
    cv.createTrackbar("sigma_r", "control", 0, 100, track_change)
    while(True):
        sigma_s = param_trackBar("sigma_s", "control")
        sigma_r = param_trackBar("sigma_r", "control", delta=0.01)

        output = cv.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)

        cv.imshow("resultado", output)
        cv.imshow("original", img)
        key = cv.waitKey(1) & 0XFF
        if(key == ord('q')):
            break
    cv.destroyAllWindows()
    return output




def defParametrosSegColor(imgBGR):
    img_HSV = cv.cvtColor(imgBGR, cv.COLOR_BGR2HSV)

    cv.namedWindow("control")
    cv.createTrackbar("Hlow","control",0,179,track_change)
    cv.createTrackbar("Hhigh","control",0,179,track_change)

    cv.createTrackbar("Slow","control",0,255,track_change)
    cv.createTrackbar("Shigh","control",0,255,track_change)

    cv.createTrackbar("Vlow","control",0,255,track_change)
    cv.createTrackbar("Vhigh","control",0,255,track_change)

    while(True):
        Hlow = param_trackBar("Hlow", "control")
        Hhigh = param_trackBar("Hhigh", "control")

        Slow = param_trackBar("Slow", "control")
        Shigh = param_trackBar("Shigh", "control")

        Vlow = param_trackBar("Vlow", "control")
        Vhigh = param_trackBar("Vhigh", "control")

        lower = np.array([Hlow, Slow, Vlow], dtype = "uint8")
        upper = np.array([Hhigh, Shigh, Vhigh], dtype = "uint8")

        img_bin = cv.inRange(img_HSV, lower, upper)
        cv.imshow("bin_color", img_bin)
        cv.imshow("original", imgBGR)

        key = cv.waitKey(1)
        if(key == ord('q')):
            break
    cv.destroyAllWindows()
    return img_bin


def filtroMediana(img, size):
    """Pareceria bueno para img con sal y pimienta y gaussiano"""
    #Puede cambiar el gris medio de la imagen
    img_f = cv.medianBlur(img, size)
    return img_f

def comp_conect(mascara):
    connectivity = 8
    # Perform the operation
    output = cv.connectedComponentsWithStats(mascara, connectivity, cv.CV_32S)
    # RESULTADOS
    # El primer elemento es el numero de etiquetas
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1] #Imagen donde cada componente conectada tiene un valor de intensidad diferente
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3] #EL CENTROIDE[0,0] ES EL DEL FONDO!!!!
    #print ('CANTIDAD DE ROSAS = ', num_labels - 1)

    return num_labels, labels, stats, centroids

def maxConvex(mascara):
    num_labels, labels, stats, centroids = comp_conect(mascara)
    Inmax=1
    AreaMax=0
    centroMax = []
    inicioX = 0
    inicioY = 0    
    alto = 0
    ancho = 0
    for x in range(1,num_labels-1):
        if(stats[x,cv.CC_STAT_AREA]>AreaMax):
            AreaMax=stats[x,cv.CC_STAT_AREA]
            Inmax=x
            centroMax = centroids[x]
            inicioX = stats[x, cv.CC_STAT_LEFT]
            inicioY = stats[x, cv.CC_STAT_TOP]
            alto = stats[x, cv.CC_STAT_HEIGHT]
            ancho = stats[x, cv.CC_STAT_WIDTH]
            
    labels[labels!=Inmax] = 0
    labels[labels==Inmax] = 255
    labels = labels.astype('uint8')
    componenteMaximaArea = labels.copy()
    
    return componenteMaximaArea, centroMax, [inicioX, inicioY, ancho, alto]

#Obtengo las coordenadas del BB a partir de las stats de componentes conectadas
def coordenadasBB(datos):
    
    x0 = datos[0]
    y0 = datos[1]
    x1 = x0 + datos[2]
    y1 = y0 + datos[3]

    return x0, y0, x1, y1

#Funcion para la mascara final, si queda algun hueco lo cierro
def retoqueCierre(mascara):
    c3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    c5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    c7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    c9 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    c11 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))

    output = cv.morphologyEx(mascara, cv.MORPH_CLOSE, c9, iterations=2)      #c9 posta
    return output


def IoU(imgGT, imgPred):
    """1: Optimo - 0: Malo"""
    imgGT = cv.cvtColor(imgGT, cv.COLOR_BGR2GRAY)

    interseccion = cv.bitwise_and(imgGT, imgPred)
    union = cv.bitwise_or(imgGT, imgPred)
    iou = cv.sumElems(interseccion)[0] / cv.sumElems(union)[0]
    print ("Intersection over Union = ", iou)
    




