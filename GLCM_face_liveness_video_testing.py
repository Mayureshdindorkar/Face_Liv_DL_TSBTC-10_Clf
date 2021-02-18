import numpy as np
import cv2
from sklearn.externals import joblib
from skimage.feature import greycomatrix, greycoprops
from skimage import io,img_as_ubyte
# from time import gmtime, strftime


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110))
    return faces


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

if __name__ == "__main__":

    # # Load model
    clf = None
    try:
        #clf = joblib.load("real_fake_rgb_replay_RandomForestClassifier.pkl")
        clf = joblib.load("real_fake_rgb_nuaa_RandomForestClassifier.pkl")
    except IOError as e:
        print("Error loading model",e)
        exit(0)
    
    #open camera
    cap = cv2.VideoCapture(int(0))

    width = 320
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # # Initialize face detector
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret, img_bgr = cap.read()
        if ret is False:
            print( "Error grabbing frame from camera")
            break

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = detect_face(img_gray, faceCascade)

        point = (0,0)
        for i, (x, y, w, h) in enumerate(faces):

            roi = img_bgr[y:y+h, x:x+w]

            for i in range(0,3):
                image = img_as_ubyte(roi[:,:,i])
                bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) 
                inds = np.digitize(image, bins)
                max_value = inds.max()+1
                matrix_coocurrence = greycomatrix(inds, [1, 2, 4, 8, 16, 32, 64, 128], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

                contrast = greycoprops(matrix_coocurrence, 'contrast')
                dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
                homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
                energy = greycoprops(matrix_coocurrence, 'energy')
                correlation = greycoprops(matrix_coocurrence, 'correlation')
                asm = greycoprops(matrix_coocurrence, 'ASM')

                contrast = contrast.ravel()
                contrast = contrast.reshape(1, len(contrast))
       
                dissimilarity = dissimilarity.ravel()
                dissimilarity = dissimilarity.reshape(1, len(dissimilarity))
       
                homogeneity = homogeneity.ravel()
                homogeneity = homogeneity.reshape(1, len(homogeneity))
               
                energy = energy.ravel()
                energy = energy.reshape(1, len(energy))
       
                correlation = correlation.ravel()
                correlation = correlation.reshape(1, len(correlation))
       
                asm = asm.ravel()
                asm = asm.reshape(1,len(asm))

                if(i == 0):
                    result = np.hstack((contrast,dissimilarity,homogeneity,energy,correlation,asm))
                else:
                    result = np.append(result, np.hstack((contrast,dissimilarity,homogeneity,energy,correlation,asm)))
                    result = result.reshape(1, len(result))                     
                      

            feature_vector = result.ravel()
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            prediction = clf.predict_proba(feature_vector)
            #print(prediction)
            #print(clf.classes_)
            #prob = clf.predict(feature_vector)

            prob=0
            if prediction[0][0]>prediction[0][1]:
                prob = 'FAKE'
            elif prediction[0][0]<=prediction[0][1]:
                prob = 'REAL'

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            point = (x, y-5)

            if prob == 'FAKE':
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text='Fake', org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)
            elif prob=='REAL':
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text='Real', org=point, fontFace=font, fontScale=0.9,
                            color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('img_rgb', img_bgr)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()