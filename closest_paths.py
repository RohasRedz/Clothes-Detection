import numpy as np  
import cv2
from joblib import dump, load

def all_distance(vector,vectores):
    # It is a matrix with the vector 'vector' repeated in all the rows
    d = np.ones((vectores.shape[0],32))
    d = d*vector

    #distances is a vector with the same number of rows as 'vectors' where each component is the distance between vector and vectors [i]
    distance = np.linalg.norm(d-vectores,axis=1)
    return distance




#class is a number, descriptors is hog_descriptors
#n is the number of closest images that we want
def get_closest_paths(descriptors,clase,n=3):
    # Descriptores
    desc = descriptors[clase][:, 1]

    # Paths
    paths_descriptors = descriptors[clase][:, 0]

    # fixing them
    wea_good = np.zeros((desc.shape[0], desc[0].shape[0]))
    for i in range(desc.shape[0]):
        wea_good[i] = desc[i]


    #final_paths = np.zeros((desc.shape[0],n))
    final_paths = []
    for i in range(desc.shape[0]):
        paths_local = []
        # Taking out the distance
        distance = all_distance(desc[i], wea_good)

        # Fixing distance
        sorted = np.sort(distance)
        for j in range(n):
            #We ignore the index 0 because it will be the same vector that we evaluate
            index = np.where(distance == sorted[j+1])[0]
            close_path = paths_descriptors[index][0]
            paths_local.append(close_path)
            print(i,j,close_path)
            #final_paths[i][j] = close_path
        print(paths_local)
        final_paths.append(paths_local)
    return final_paths

def get_hog(image,coord=(0,0,0,0),size=(80,150)):
    hog = cv2.HOGDescriptor()
    x1,y1,x2,y2 = coord[0], coord[1], coord[2], coord[3]

    #If w, h came at zero, there is no need to trim the image
    # if (w==0):
    #     w = image.shape[1]
    # if(h==0):
    #     h = image.shape[0]

    #recreate
    image = image[y1:y2, x1:x2]
    #resize
    image = cv2.resize(image,size)
    hog_im = hog.compute(image)

    return hog_im

def n_paths_close(vector,descriptores,clase,n=3):
    # Descriptores
    desc = descriptores[clase][:, 1]
    #print(desc.shape)
    # Paths
    paths_descriptors = descriptores[clase][:, 0]
    #prnt(len(descriptores[clase]))
    #print(descriptores[clase].shape)
    #fixing
    wea_good = np.zeros((desc.shape[0],32))
    #print(wea_good.shape)
    for i in range(desc.shape[0]):
        wea_good[i] = desc[i]

    # taking out distance
    distance = all_distance(vector, wea_good)

    # fixing 
    sorted = np.sort(distance)

    closest_paths = []
    for i in range(n):
        index = np.where(distance == sorted[i+1])[0] #ignore index 0
        close_path = paths_descriptors[index]
        closest_paths.append(close_path[0])
        
        
    return closest_paths




if __name__ == "__main__":

    pca_objs = {}
    hog_descriptors = {}
    i=0
    classes = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
    for clss in classes:
        pca_objs[i] =  load('pca_objs/{}{}'.format(clss,'.joblib'))
        hog_descriptors[i] = np.load('hog_descriptors/{}{}'.format(clss,'.npy'), allow_pickle = True)
        i+=1


    pca = pca_objs[int(0)]
    hog_detection = np.ones(34020) #get_hog(frame,(x1, y1, x2, y2))
    hog_pca = pca.transform(hog_detection.reshape(1, -1))
    paths = n_paths_close(hog_pca, hog_descriptors,0,n=1)
    print(paths)

