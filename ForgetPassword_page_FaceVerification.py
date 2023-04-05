
""" Goal: Gathering all the stages in one file for the clean implementation """

"""Process:
1. Prepare MTCNN function
2. Prepare Embeddings function
3. Merge them all and reshape the embedded feature
"""

""" Libraries """
import Extract_Face_Mtcnn_stage01
import Embedded_Features_VggNet_stage02

# Prepare cropped image using MTCNN
def prepare_mtcnn(image_path, bool_val):

    x_cropped_face, zipped_coord = Extract_Face_Mtcnn_stage01.extract_face(image_path, bool_val)

    return x_cropped_face, zipped_coord

# Prepare facial embeddings using vgg_model
def prepare_Embeddings(cropped_face):

    Cropped_img = cropped_face[0]

    # VGG model based on ResNet50
    model = Embedded_Features_VggNet_stage02.vgg_model_building()

    # emebedded feature of the cropped_face
    embedded_face = Embedded_Features_VggNet_stage02.embedded_feature(model, Cropped_img)

    return embedded_face

# get the current user data opted for forget password
def currentUser_ImageProcess(ImageArray):

    # Sends the image array for MTCNN -> Embedded
    # 1. MTCNN
    cropped_face, zipped_corrd = prepare_mtcnn(ImageArray, bool_val=True)

    # 2, Embeddings
    embedded_face = prepare_Embeddings(cropped_face)

    currentUser_embedded_face = embedded_face[0, :]  # [0,:] -> to bring the shape from (1,2048) to (2048,)

    return currentUser_embedded_face




