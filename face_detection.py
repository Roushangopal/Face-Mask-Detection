import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab


def getEncodingsOfDatasets():
    # getting all the images from our dataset to search for the names
    path = 'images'
    images = []
    names = []
    myList = os.listdir(path)
    # print(myList)
    print("[INFO] Fetching Datsets!")
    # use the names of the images and import one by one
    for imgName in myList:
        curImg = cv2.imread(f'{path}/{imgName}')
        images.append(curImg)
        names.append(os.path.splitext(imgName)[0])
    print("[INFO] Fetching Completed!")
    # print(names)
    encodeListKnown = findEncodings(images)
    print("[INFO] Encoding Completed!")
    return [encodeListKnown, names]

# def markWithoutMask(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#         entry = line.split(',')
#         nameList.append(entry[0])
#         if name not in nameList:
#         now = datetime.now()
#         dtString = now.strftime('%H:%M:%S')
#         f.writelines(f'n{name},{dtString}')


# find the encodings of each image
def findEncodings(images):
    print("[INFO] Encoding Started!")
    encodeList = []
    for img in images:
        # converted to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # finding the encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


if __name__ == '__main__':
    output = getEncodingsOfDatasets()
    encodeListKnown = output[0]
    names = output[1]
    print("[INFO] Starting Webcam")
    cap = cv2.VideoCapture(0)
    # reading each frame one by one
    while True:
        success, img = cap.read()
        # reduce the size of image for speed
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # convert into RGB
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
        # we can get multiple faces in a frame so we have to find the location
        facesCurFrame = face_recognition.face_locations(imgSmall)
        # find the encoding
        encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
        # find the matches between our encodings
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            # finding lowest element in our list for best match
            matchIndex = np.argmin(faceDis)
            # Display a rectangle around the face with name
            if matches[matchIndex]:
                name = names[matchIndex].upper()
                print(name)


        # y1, x2, y2, x1 = faceLoc
        # # we have scaled our input to 0.25 so in order to revive original value we have multiplied by 4
        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        # # creating the rectangle
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # # rectangle for name
        # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        # # name
        # cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        # markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)