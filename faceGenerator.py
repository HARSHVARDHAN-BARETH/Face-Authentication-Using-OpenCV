

"""
    Requirements
    python 3.7.7
    numpy==1.21.6
    opencv-contrib-python==4.1.2.30
    Pillow==9.5.0
"""

import sys
import time
import os
import numpy as np
from PIL import Image
import cv2
path = 'user_data'
name =''
if not os.path.exists("user_data"):
    os.mkdir('user_data')
    # print("Directory " , dirName ,  " Created ")


def face_generator():
    global name
    cam = cv2.VideoCapture(0)  # used to create video which is used to capture images
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    
    cam.set(3, 640)
    cam.set(4, 480)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_id = input("Enter id of user: ")
    name = input("Enter name: ")
    sample = int(input("Enter how many samples you wish to take: "))

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    print("Taking sample image of user... Please look at the camera.")
    time.sleep(4)  # Increased delay to 4 seconds

    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(converted_image, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"user_data/face.{face_id}.{count}.jpg", converted_image[y:y + h, x:x + w])
            cv2.imshow("image", img)

        k = cv2.waitKey(1) & 0xff
        if k == 27 or count >= sample:
            break

    print("Image samples taken successfully!")
    cam.release()
    cv2.destroyAllWindows()





def traning_data():
    # Initialize LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load the Haar Cascade Classifier for face detection
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def load_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            gray_image = Image.open(image_path).convert('L')
            img_arr = np.array(gray_image, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])

            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)
        return face_samples, ids

    print("Training Data...please wait...!!!")
    faces, ids = load_images_and_labels(path)

    recognizer.train(faces, np.array(ids))
    recognizer.save('trained_data.yml') 

    print("Data trained successfully!")






def detection():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_data.yml')  # loaded trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes font size

    id = 0  # Initialize id to a default value
    names = ['', name]  # Ensure at least two elements in the names list
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try using DirectShow backend
    cam.set(3, 640)  # set camera resolution
    cam.set(4, 480)

    if not cam.isOpened():
        print("Error: Failed to open camera.")
        return

    minW = 0.1 * cam.get(3)  # define min window size to be recognized as a face
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()  # read the frames using above created objects
        if not ret:
            print("Error: Failed to capture image from camera")
            break

        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts image to grayscale
        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])

            if accuracy < 100:
                id = names[id] if id < len(names) else "unknown"
                accuracy = " {0}%".format(round(100 - accuracy))
            else:
                id = "unknown"
                accuracy = " {0}%".format(round(100 - accuracy))

            cv2.putText(img, "Press Esc to close this window", (5, 25), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 0, 255), 2)
            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def permisssion(val,task):
    if "Y"==val or "y" ==val:
         if task == 1:
             traning_data()
         elif task == 2:
             detection()
    else:
        print("ThankYou for using this application !! ")
        sys.exit()


print("\t\t\t  Welcome to face Authentication System ")
face_generator()
perm=input("Do you wish to train your image data for face authentication [y|n] : ")
permisssion(perm,1)
authenticate=input("Do your want test authentication system [y|n] : ")
permisssion(authenticate,3)

