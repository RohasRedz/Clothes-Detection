import numpy as np
import cv2
from sklearn.cluster import KMeans



def colores_dominantes(img, n_clusters):

    #The images with opencv by default
    #open in bgr. We passed them to lab to saw
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #We went to the list to see it
    img = img.reshape((img.shape[0] * img.shape[1], 3))


    # We use kmeans to saw it with the n_clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(img)

    #The clusters
    clusters = kmeans.cluster_centers_

    return clusters




#Receive a tuple (l, a, b), transform to hsv and determine the name of the color
def check_color_hsv(lab):
    l,a,b = lab[0],lab[1],lab[2]

    #We create a 1-pixel 3-channel image
    array = np.array([l,a,b],dtype=np.uint8)
    array = np.reshape(array,(1,1,3))

    #Transform to hsv space
    bgr = cv2.cvtColor(array,cv2.COLOR_LAB2BGR) #Q xuxa no hay lab2hsv
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)

    #h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]

    h, s, v = hsv[0, 0, 0], hsv[0, 0, 1], hsv[0, 0, 2]
    print(h,s,v)
    if s<25:
        if v<85:
            return 'Black'
        elif v<170:
            return 'Gray'
        return 'White'

    if h<=5 or h>175:
        return 'Red'

    if h>5 and h<=15:
        return 'orange'

    if h>15 and h<=40:
        return 'Yellow'

    if h>40 and h<=70:
        return 'Green'

    if h>70 and h<=105:
        return 'Light Blue'

    if h>105 and h<=130:
        return 'Blue'

    if h>130 and h<=175:
        return 'Dwelling'


def color_imagen(imagen,n_clusters=3):
    color_lab_dom = colores_dominantes(imagen,n_clusters)
    l,a,b = color_lab_dom[0][0], color_lab_dom[0][1], color_lab_dom[0][2]

    number_color = check_color_hsv((l,a,b))

    return number_color

