# Imports
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import math
import argparse
from pathlib import Path
import os
import ntpath
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import streamlit as st
import tempfile

#title of the App
st.title("SISE facial recognition")

try :
    f = st.file_uploader("Choose a Video")     
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
    st.markdown('[INFO] video importée') #'#import de la base de connaisance (import des SISE) ...')
    st.markdown('[INFO] Lancement de l\'application...')   
except :
    st.markdown("[INFO] Pas de vidéo")


# Pretrained models
# Identify the locations of important facial landmarks
pose_predictor_68_point = dlib.shape_predictor("C:/Users/adrien/Downloads/BAROU_DECOSTER_PAVOINE/face_detection/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("C:/Users/adrien/Downloads/BAROU_DECOSTER_PAVOINE/face_detection/shape_predictor_5_face_landmarks.dat")
# Maps human faces into vectors where pictures of the same person are mapped near to each
face_encoder = dlib.face_recognition_model_v1("C:/Users/adrien/Downloads/BAROU_DECOSTER_PAVOINE/face_detection/dlib_face_recognition_resnet_model_v1.dat")
# Face detector
face_detector = dlib.get_frontal_face_detector()



def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image) # Face detection
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1))) # Compute vector which describe the face
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape) # Points which allows to calculate the coord of the face
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list

# Function for face recognition
def easy_face_reco(frame, known_face_encodings, known_face_names): 
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.5
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left + 2, bottom + 21), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')


args=parser.parse_args()

# Initialize the face classifier for emotion
face_classifier = cv2.CascadeClassifier('./emotion_detection/haarcascade_frontalface_default.xml')
classifier =load_model('./emotion_detection/Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

#st.markdown('[INFO] Lancement de l\'application...')   
face_to_encode_path = Path('./known_faces')#args.input)
files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

for file_ in face_to_encode_path.rglob('*.png'):
    files.append(file_)
if len(files)==0:
    raise ValueError('No faces detect in the directory: {}'.format(face_to_encode_path))
known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

known_face_encodings = []
for file_ in files:
    image = PIL.Image.open(file_)
    image = np.array(image)
    face_encoded = encode_face(image)[0][0]
    known_face_encodings.append(face_encoded)




# Models for age and gender recognition 
faceProto = "./age_gender_detection/opencv_face_detector.pbtxt"
faceModel = "./age_gender_detection/opencv_face_detector_uint8.pb"
ageProto = "./age_gender_detection/age_deploy.prototxt"
ageModel = "./age_gender_detection/age_net.caffemodel"
genderProto = "./age_gender_detection/gender_deploy.prototxt"
genderModel = "./age_gender_detection/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-8)', '(10-16)', '(18-22)', '(24-32)', '(38-48)', '(48-60)', '(64-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)





video = cv2.VideoCapture(tfile.name)

image_placeholder = st.empty()
padding=20

fps = int(video.get(cv2.CAP_PROP_FPS))
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')



out = cv2.VideoWriter('./result/output.avi', fourcc, fps-14, (w,h))

while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    
    labels = []
    try:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # remove the color
    except:
        break
            
    
    faces = face_classifier.detectMultiScale(gray,1.3,5) # detect the faces

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0] # predict the probability of each particular class
            label=class_labels[preds.argmax()] # index of maximum probability
            label_position = (x,y-64)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    easy_face_reco(resultImg, known_face_encodings, known_face_names)
    
    #Age and gender recognition
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    out.write(resultImg) 
    
    image_placeholder.image(resultImg, channels="BGR")


out.release()
st.markdown("[INFO] Vidéo sauvegardé ...")
st.markdown('[INFO] FIN !')