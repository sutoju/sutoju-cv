import sys
import os
ROBOT = False
if len(sys.argv) > 1 and sys.argv[1] == 'robot':
    ROBOT = True

if ROBOT:
    from PIL import Image

from naoqi import ALProxy
import numpy as np
import cv2
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
import time
import thread
import requests
import json

WATSON_KEY = os.environ['WATSON_KEY']
ENDPOINT_PASS = os.environ['ENDPOINT_PASS']

classes = json.loads(open('./classes.json', 'r').read())
predefined_classes = [954, 968, 950, 939, 943, 957]
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')

if not ROBOT:
    cap = cv2.VideoCapture(0)

TIMEOUT = 0

def tunkkaa_apia(food):
    url = 'https://sutoju-backend.eu-gb.mybluemix.net/food/' + str(food)
    r = requests.post(url, data={'password': ENDPOINT_PASS})
    print (r.text)
    return None

def classify_image(filepath, api_key):
    predefined_classes = ['banana', 'apple', 'cucumber']
    files = open(filepath,'rb').read()
    for i in range(len(keys)):
        url = 'https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key='+WATSON_KEY+'&version=2016-05-20'
        r = requests.post(url, data=files, verify=False)
        j = json.loads(r.text)
        if 'status' in j and j['status'] == 'ERROR':
            keys.pop(0)
            continue
        try:
            classes = j['images'][0]['classifiers'][0]['classes']
            classes.sort(key=lambda x: x['score'], reverse=True)

            for i in classes:
                if i['class'] in predefined_classes:
                    return i['class']
            
            #return classes[0]['class']
        except Exception as e:
            print(e)

    return None

if ROBOT:
    proxy = ALProxy('ALVideoDevice', '192.168.1.19', 9559)
    videoClient = proxy.subscribe('python_client', 1, 11, 5)
tts = ALProxy('ALTextToSpeech', '192.168.1.19', 9559)
tts.setLanguage('English')
led = ALProxy('ALLeds', '192.168.1.19', 9559)

def read_item_name(name):
    tts.say("One, {}".format(str(classes[c_id])))
def rotate_eyes():
    led.rotateEyes(0x0000ff00, 0.3, 1.0)

fgbg = cv2.createBackgroundSubtractorMOG2()


while (True):
    if ROBOT:
        frame = proxy.getImageRemote(videoClient)
        proxy.releaseImage(videoClient)
        width = frame[0]
        height = frame[1]
        frame = Image.frombytes('RGB', (width, height), frame[6])
        opencvImage = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    else:
        ret, opencvImage = cap.read()

    frame = np.copy(opencvImage)
    #mask = cv2.cvtColor(np.copy(opencvImage), cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(mask,(7,7),0)
    #ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #im2,contours, hierarchy = cv2.findContours(np.copy(otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = [c for c in np.copy(contours) if cv2.contourArea(c) > 10000]
    if TIMEOUT < 1:
        ''' 
        cnt = contours[0]
        largest = 0
        for c in contours:
            if cv2.contourArea(c) > largest:
                largest = cv2.contourArea(c)
                cnt = c
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       ''' 
        cropped = np.copy(frame)#[y:y+h,x:x+w]
        frame2 = cv2.resize(cropped, (227,227))
        frame2 = frame2[...,::-1]
        frame2 = np.swapaxes(frame2, 0,2)
        frame2 = np.swapaxes(frame2, 1,2)
        frame2 = np.array([frame2])
        pred = model.predict(frame2)[0]
        prob = [pred[i] for i in predefined_classes]
        c_id = np.argmax(prob)
        c_id = predefined_classes[c_id]
        if pred[c_id] > 0.01:
            print(classes[c_id], prob)
            thread.start_new_thread(rotate_eyes, ())
            thread.start_new_thread(read_item_name, (classes[c_id],))
            tunkkaa_apia(classes[c_id])
            TIMEOUT = 5

    if TIMEOUT > 0:
        TIMEOUT -= 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
