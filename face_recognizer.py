#!/usr/bin/python

# Import the required modules
import cv2, os, sys
import numpy as np

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()
fisher_recogniser = cv2.face.createFisherFaceRecognizer()
eigen_recogniser = cv2.face.createEigenFaceRecognizer()


def get_images_and_labels(path):
    print path
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.wink.png') and not f.endswith('.DS_Store')]
    # images will contains face images
    images = []
    images_resized = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(gray[y: y + h, x: x + w])
            resized = cv2.resize(gray[y: y + h, x: x + w], (200, 200))
            images_resized.append(resized)
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", gray[y: y + h, x: x + w])
            cv2.waitKey(1)
    # return the images list and labels list
    return images, images_resized, labels

# Path to the Yale Dataset
path = './yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, images_resized, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))
fisher_recogniser.train(images_resized, np.array(labels))
eigen_recogniser.train(images_resized, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wink.png')]
for image_path in image_paths:
    predict_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = faceCascade.detectMultiScale(
        predict_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        nbr_predicted, confidence = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized by LBPH with confidence {}".format(nbr_actual, confidence)
        else:
            print "{} is Incorrect Recognized by LBPH as {}, with confidence {}".format(nbr_actual, nbr_predicted, confidence)

        resized_image = cv2.resize(predict_image[y: y + h, x: x + w], (200, 200))
        nbr_predicted, confidence = fisher_recogniser.predict(resized_image)
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized by Fisher with confidence {}".format(nbr_actual, confidence)
        else:
            print "{} is Incorrect Recognized by Fisher as {}, with confidence {}".format(nbr_actual, nbr_predicted, confidence)

        nbr_predicted, confidence = eigen_recogniser.predict(resized_image)
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized by Eigen with confidence {}".format(nbr_actual, confidence)
        else:
            print "{} is Incorrect Recognized by Eigen as {}, with confidence {}".format(nbr_actual, nbr_predicted, confidence)


        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(10)

video_capture = cv2.VideoCapture(0)


while True:
    try:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            nbr_predicted, confidence = recognizer.predict(gray[y: y + h, x: x + w])
            resized = cv2.resize(gray[y: y + h, x: x + w], (200, 200))

            # print "{} is Recognized with confidence unknown".format(nbr_predicted)
            i_conf = int(round(confidence))
            if i_conf < 0:
                i_conf = 0
            if i_conf > 255:
                i_conf = 255
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255-i_conf, i_conf), 2)
            cv2.putText(frame, "LBPH {}".format(nbr_predicted), (x+5, y+20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, "{}".format(confidence), (x+5, y+35), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

            nbr_predicted, confidence = fisher_recogniser.predict(resized)
            cv2.putText(frame, "Fisher {}".format(nbr_predicted), (x+5, y+60), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, "{}".format(confidence), (x+5, y+75), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

            nbr_predicted, confidence = eigen_recogniser.predict(resized)
            cv2.putText(frame, "Eigen {}".format(nbr_predicted), (x+5, y+100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, "{}".format(confidence), (x+5, y+115), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)


        # Display the resulting frame
        cv2.imshow('Video', frame)

        key = cv2.waitKey(2)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            user_input = raw_input("Face Number: ")
            nbr = int(user_input)
            update_images = []
            update_labels = []
            for (x, y, w, h) in faces:
                update_images.append(gray[y: y + h, x: x + w])
                resized = cv2.resize(gray[y: y + h, x: x + w], (200, 200))

                images_resized.append(resized)

                labels.append(nbr)
                update_labels.append(nbr)
                cv2.imshow("Adding faces to traning set...", gray[y: y + h, x: x + w])
                cv2.waitKey(1)

            recognizer.update(update_images, np.array(update_labels))
            fisher_recogniser.train(images_resized, np.array(labels))
            eigen_recogniser.train(images_resized, np.array(labels))
    except:
        e = sys.exc_info()[0]
        print e

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
