import numpy as np
import cv2
from sys import exit

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from tensorflow.keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

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

    #model = tf.keras.models.load_model('vgg19_nuaa_0_00001_4CB_D.h5')
    #model = tf.keras.models.load_model('VGG16_try_models/VGG16_0_0001_FL_best_model.h5') #works good but keep distant for video attack
    model = tf.keras.models.load_model('VGG19_try_models/VGG19_0_0001_FL_best_model.h5') #works good for all
    model.summary()
    
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
            
            #new_roi = cv2.resize(roi,(96,96))
            new_roi = cv2.resize(roi,(224,224)) #96,96
            #print("NEW ROI : ",type(new_roi)," ",new_roi.shape) # numpy.ndarray (150,150,3)
            
            imgArray = img_to_array(new_roi)
            
            #imgArray = imgArray.reshape(1,96,96,3)
            imgArray = imgArray.reshape(1,224,224,3) #(1,96,96,3) 
            
            imgArray = imgArray/float(255)
            
            #print("imgArray : ",type(imgArray)," ",imgArray.shape) #numpy.ndarray (1,150,150,3)
            
            #Real==1 Fake==0
            #guess = int(model.predict_classes(imgArray,verbose=0))
            guess = model.predict(imgArray)
            guess = int(round(guess[0][0])) #guess<=0.5 -> 0 and guess > 0.5 -> 1

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            point = (x,y-5)
            
            #if(guess == 0):
            #    print (guess," Guess by model : Fake")
            #elif(guess == 1):
            #    print (guess," Guess by model : Real")
                
            if(guess == 0):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text="Fake Attempt", org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text="Real Attempt", org=point, fontFace=font, fontScale=0.9,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            

        cv2.imshow('img_rgb', img_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()