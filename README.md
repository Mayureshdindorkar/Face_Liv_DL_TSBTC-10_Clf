# Face Liveness Detection (Face Anti-spoofing):  

- Built an AI model that determines whether the face of the person sitting in front of the webcam is Real or Fake by using Transfer Learning and TSBTC (Thepade’s Sorted Bock Truncation Encoding).

- The model consistes of two parts:
    1. Feature extractor:  
        a. Deep learning based feature extractor: Finetuned VGG19  
        b. Handcrafted feature extractor        : TSBTC-10 ary (Thepade's Sorted Block Truncation Encoding)  
    3. Classifier:  
         Decision Tree    

- Created model is robust to Photo attack, video attack, warped photo attack, etc.

---   
\
**Steps to run the code**:  

1. Install the necessary dependancies:
```
pip install -r requirements.txt
```

2. Run the file:
```
python DL_TSBTC_clf_video_testing.py
```
\
The corresponding research papers are: [paper](https://www.ije.ir/article_161610.html)  
```
Thepade, S. D., Dindorkar, M. R., Chaudhari, P. R., & Bang, S. V.  (2023). Enhanced Face Presentation Attack Prevention Employing Feature Fusion of Pre-trained Deep Convolutional Neural Network Model and Thepade's Sorted Block Truncation Coding. International Journal of Engineering, 36(4), 807-816. doi: 10.5829/ije.2023.36.04a.17
```
