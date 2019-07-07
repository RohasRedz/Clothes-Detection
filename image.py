
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
from closest_paths import n_paths_close, get_hog
from joblib import dump, load
import colores

def load_model(params):
# Set up model
    model = Darknet(params['model_def'], img_size=params['img_size']).to(params['device'])
    model.load_darknet_weights(params['weights_path'])
    model.eval()  # Set in evaluation mode
    return model

def cv_img_to_tensor(img, dim = (416, 416)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1))
    x = x.unsqueeze(0).float()     
    _, _, h, w = x.size()
    ih, iw = dim[0],dim[1]
    dim_diff = np.abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
    x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
    x = F.upsample(x, size=(ih, iw), mode='bilinear',align_corners=True) # x = (1, 3, 416, 416)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {   "model_def" : "df2cfg/yolov3-df2.cfg",
"weights_path" : "weights/yolov3-df2_10000.weights",
"class_path":"df2cfg/df2.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device
}


classes = load_classes(params['class_path']) 
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
cmap = plt.get_cmap("tab20b")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)

model = load_model(params)

#cap = cv2.VideoCapture(0)



pca_objs = {}
hog_descriptors = {}
i=0
for clss in classes:
    pca_objs[i] =  load('pca_objs/{}{}'.format(clss,'.joblib'))
    hog_descriptors[i] = np.load('hog_descriptors/{}{}'.format(clss,'.npy'), allow_pickle = True)
    i+=1


print('Descriptors loaded')


cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Retrieval', cv2.WINDOW_NORMAL)

while(True):
    img_path = input('input file path: ')
    if img_path=='exit':
        break

    frame = cv2.imread(img_path)
    if frame is None:
        print('Image not found.')
        continue
    
    img= frame.copy()     
    x = cv_img_to_tensor(img)
    x.to(device)
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Get detections
    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])
    #closest_img = None
    if detections[0] is not None:

        # Rescale boxes to original image
            
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds , seed)
        bbox_colors = colors[:n_cls_preds]
        closest_img_paths = []
        colors = colores.color_imagen(img)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            print("color: %s"% (colors))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            
            color = tuple(c*255 for c in color)
            color = (0,0,0)
            #color = (color[2],color[1],color[0])

            
            cv2.rectangle(frame,(x1,y1) , (x2,y2) , color,3)
            #print(int(cls_pred))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf.item())
            text2 = "color : %s"%(colors)
            cv2.rectangle(frame,(x1-2,y1-25) , (x1 + 8.5*len(text),y1) , color,-1)
            cv2.putText(frame,text,(x1,y1-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,text2,(x1,y1+25), font, 0.5,(25,80,90),1,cv2.LINE_AA)
            try:
                pca = pca_objs[int(cls_pred)]
                hog_detection = get_hog(frame,(int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))
                hog_pca = pca.transform(hog_detection.reshape(1, -1))
                closest_img = n_paths_cercanos(hog_pca,hog_descriptors,int(cls_pred),n=1)
                closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))
            except:
                continue
            
             
    
        if(len(closest_img_paths)>=1):
            for im_path in closest_img_paths:
                    img_retrieval = cv2.imread(im_path[0])
                    cv2.imshow(im_path[0],img_retrieval)
    cv2.imwrite('output/image.png',frame)
    cv2.imshow('Detections',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
                
        
