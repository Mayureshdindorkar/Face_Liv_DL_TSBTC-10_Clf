import numpy as np
import cv2
from sys import exit
import pickle
import tensorflow as tf
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Input
#from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.preprocessing.image import img_to_array

#import cv2
import statistics
#import xlsxwriter

finetuned_model = tf.keras.models.load_model('VGG19_try_models/VGG19_0_0001_FL_best_model.h5')

#x = finetuned_model.get_layer('dense_4').output    # for VGG16
x = finetuned_model.get_layer('dense').output     # for VGG19

#features = Flatten()(x)                #for vgg16
features = Flatten(name='flatten1')(x)  #for vgg19

feature_ext_model = Model(inputs = finetuned_model.input, outputs = features)
feature_ext_model.summary()

# Decision Tree model
loaded_model = pickle.load(open('VGG19_try_models/VGG19+TSBTC+DT_512_30.pkl', 'rb'))  #works very well. Just dont show salakha's photo.
def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110))
    return faces

#################################
def TSBTC( BTCLevel, noOfDigitsAfterDecimal, input_image):
    
    #totalClasses = len(category)    
    
    #count = 2 #excel sheet cell number (from where to write the respective feature vector)
    
    #workbook = xlsxwriter.Workbook(pathForResultFileStorage)
    #worksheet = workbook.add_worksheet()
    
    #for i in range(1, totalRecords + 1):        
    #pathOfRecord = pathOfRecords + str(i) + image_extension
    #print(str(i) + image_extension) #just to know which image is currently going on
    x = input_image    
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    r, c, pl = x.shape

    min = 0
    max = c
    tfvred = [0] * (r * c)
    tfvgreen = [0] * (r * c)
    tfvblue = [0] * (r * c)

    for xi in range(0, r):
        tfvred[min : max] = x[xi, 0 : c, 0]
        tfvgreen[min : max] = x[xi, 0 : c, 1]
        tfvblue[min: max] = x[xi, 0: c, 2]
        min = max
        max = max + c

    tfvred.sort()
    tfvgreen.sort()
    tfvblue.sort()

    tfvred = [float(z) for z in tfvred]
    tfvgreen = [float(z) for z in tfvgreen]
    tfvblue = [float(z) for z in tfvblue]
    
    fvred = [0] * BTCLevel
    fvgreen = [0] * BTCLevel
    fvblue = [0] * BTCLevel

    vmin = 0
    vmax = (r * c) / BTCLevel

    for xxj in range(1, BTCLevel + 1):
        vmax = int((xxj * r * c) / BTCLevel)
        fvred[xxj - 1] = round(statistics.mean(tfvred[vmin:vmax+1]), noOfDigitsAfterDecimal)
        fvgreen[xxj - 1] = round(statistics.mean(tfvgreen[vmin:vmax+1]), noOfDigitsAfterDecimal)
        fvblue[xxj - 1] = round(statistics.mean(tfvblue[vmin:vmax+1]), noOfDigitsAfterDecimal)
        vmin = vmax
    sizefv = BTCLevel * 3
    fv = fvred + fvgreen + fvblue
    return fv

    #categoryClass = ""
    #for xk  in range(0, len(category)):
    #    if i >= minclass[xk] and i <= maxclass[xk]:
    #        categoryClass = category[xk]
    #        break

    #positionForFV = 'A' + str(count)
    #fv = tuple(fv)
    #worksheet.write_row(positionForFV, fv)
    #positionForClassName = classNameCellAlphabet + str(count)
    #worksheet.write(positionForClassName, categoryClass)

    #count = count + 1
    #workbook.close()
#######################################


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
            features = feature_ext_model.predict(imgArray)   #VGG19 (256)
            
            ############ start tsbtc #######
            #minclass = [1]
            #maxclass = [100]
            #category = ['land']
            BTCLevel = 10
            #pathOfRecords = "C:\\Users\\Piyush\\PycharmProjects\\TSBTCInPython\\"
            #pathForResultFileStorage = "C:\\Users\\Piyush\\PycharmProjects\\TSBTCInPython\\mamu.xlsx"
            noOfDigitsAfterDecimal = 5
            #classNameCellAlphabet = 'Z'
            #image_extension = '.tif'
            totalRecords = 5
            tsbtcfv = TSBTC(BTCLevel, noOfDigitsAfterDecimal, new_roi)
            
            ############ end tsbtc ########
            features = features[0].ravel()  #numpy.ndarray
            features = np.concatenate([features,features]) # (DL=512, TSBTC=30)

            tsbtcfv = np.array(tsbtcfv)
            z = np.concatenate([features,tsbtcfv])
            z = z.tolist()
            
            guess = loaded_model.predict([z])   #SVM

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