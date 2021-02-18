import numpy as np
import cv2
from sys import exit
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from tensorflow.keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

finetuned_model = tf.keras.models.load_model('VGG19_try_models/VGG19_0_0001_FL_best_model.h5')

#x = finetuned_model.get_layer('dense_4').output    # for VGG16
x = finetuned_model.get_layer('dense').output     # for VGG19

#features = Flatten()(x)                #for vgg16
features = Flatten(name='flatten1')(x)  #for vgg19

feature_ext_model = Model(inputs = finetuned_model.input, outputs = features)
feature_ext_model.summary()

#loaded_model = pickle.load(open('VGG19_try_models/VGG19+SVM.pkl', 'rb'))  #works very well. Just dont show salakha's photo.
loaded_model = pickle.load(open('VGG19_try_models/VGG19+KNN.pkl', 'rb'))  #works very well. Just dont show salakha's photo.

def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110))
    return faces


if __name__ == "__main__":
    
    cap = cv2.VideoCapture(int(0))
    
    if not cap.isOpened():
        print("Error in opening camera!")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)
        
    width = 320
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    # # Initialize face detector
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret, img_bgr = cap.read()
        if ret is False:
            print( "Error grabbing frame from camera!")
            cap.release()
            cv2.destroyAllWindows()
            break

        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        faces = detect_face(img_gray, faceCascade)

        point = (0,0)
        for i, (x, y, w, h) in enumerate(faces):

            roi = img_bgr[y:y+h, x:x+w]
            #print("ROI : ",type(roi)," ",roi.shape) #numpy.ndarray (115,115,3)
            
            new_roi = cv2.resize(roi,(224,224)) #96,96
            #print("NEW ROI : ",type(new_roi)," ",new_roi.shape) # numpy.ndarray (150,150,3)
            
            imgArray = img_to_array(new_roi)
            imgArray = imgArray.reshape(1,224,224,3) #((1,96,96,3))
            imgArray = imgArray/float(255)
            
            #print("imgArray : ",type(imgArray)," ",imgArray.shape) #numpy.ndarray (1,150,150,3)
            
            #Real==1 Fake==0
            #guess = int(model.predict_classes(imgArray,verbose=0))
            features = feature_ext_model.predict(imgArray)  #VGG19 (256)
            
            guess = loaded_model.predict(features)          #SVM

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            point = (x,y-5)
                
            if(guess == 'Fake'):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text=guess[0], org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text=guess[0], org=point, fontFace=font, fontScale=0.9,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            

        cv2.imshow('img_rgb', img_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()