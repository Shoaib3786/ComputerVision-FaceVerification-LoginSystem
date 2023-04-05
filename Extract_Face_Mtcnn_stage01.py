""" Goal: """
# Extract Face using MTCNN from Images

""" Libraries """
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

x1, y1, x2, y2 =0, 0, 0, 0

# detected face from image
def extract_face_n_labels(mImage, imageIsArray=True):

    # using MTCNN for face detector in an image
    detector = MTCNN()

    cropped_face = []

    global x1, y1, x2, y2

    if imageIsArray == True:

        pixels = mImage

        # MTCNN for face detection
        face_detected = detector.detect_faces(pixels)
        print('face_detected, proccessed values: ', face_detected)

        if face_detected:  # if face_detected list not empty

            # getting the bounding box of detected face
            x1, y1, w, h = face_detected[0]['box']

            x1, y1 = abs(x1), abs(y1)
            x2 = abs(x1 + w)
            y2 = abs(y1 + h)

            # get face from the image by slicing out using coordinates & store it
            store_face = pixels[y1:y2, x1:x2]  # y -> rows, x -> columns

            # for verification plotting the face
            # plt.imshow(store_face)

            image1 = Image.fromarray(store_face, 'RGB')  # convert the numpy array to object
            image1 = image1.resize((224, 224))  # resize the image
            face_array = np.array(image1)  # image1 to numpy array

            # increase the dim as VGGnet needs 4d
            face_array = np.expand_dims(face_array, axis=0)

            # get list of all numpy face arrays
            cropped_face.append(face_array)

    zipped_coord = [x1,y1,x2,y2]

    # return cropped_face, array_img_labels
    return cropped_face, zipped_coord