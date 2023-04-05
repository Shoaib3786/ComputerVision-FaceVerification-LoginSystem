
""" Goal: Run Flask to:
  1. Add new User into our database (Signup Page)
  2. Build Login Page (verify current user email_id + if password not remember switch to forget passwprd page)
  3. Build Forget Password Page (Use Facial verification in forget password)


## Need to save 3 things into the database:
 - Email_ID
 - Password
 - Facial Embedded feature of the new user

"""


"""Process:

1. Database connectivity

2. Signup Page:
    # Takes input 
        - Email ID
        - Password
        - Image (Live stream using webcam)
    ** Finally Saves all the 3 thing into the database. 

3. Login Page:
    # Takes input
        - Email
        - Password (If not remember switch to forget pass page)
        - Before switching verify whether email_id is existing in Db, if not then
          current user goes back to the Signup Page.
        - If current user email id verified then fetch its appropriate Face embedding from database using its email id as unique identifiers for searching.

4. Forget Password page:
    # Takes input:
        - Image(From live streaming Face Detection [used Haar Cascade])
        - Calculate Current User Face Embedding. 
        - Match the Db Embedding and Current User Embedding 
            If, matches then go to Home page.
            else, back to login page

"""

""" Libraries """
import cv2
import numpy as np
import ForgetPassword_page_FaceVerification
import firebase_admin
from firebase_admin import credentials, db
import yaml
import json
from json import JSONEncoder
from flask import Flask, render_template, request, Response, redirect, url_for, flash
from PIL import Image
import Face_Verifiction_stage03
import threading
import time

"""Getting Config file"""
with open('config.yml', 'r') as file:
    config_data = yaml.safe_load(file)

app = Flask(__name__)
app.secret_key = "super secret key"
# for displaying msg for custom duration after successful signup
app.config['MESSAGE_FLASHING_OPTIONS'] = {'duration': 100}

"""Global Variables"""
capture = 0
mcapture = 0
error = None
successfulMsg = None

ImageArray = np.array([])
imgArray = np.array([])

dbUser_embedding = np.array([])
currentUser_embedding = np.array([])

euclid_distance = 0
cosine_similarity = 0
threshold_cosine = 0
threshold_euclid = 0

# declaring lock
forgetPage_lock = threading.Lock()
mainThread_lock = threading.Lock()

cam = cv2.VideoCapture(2)


# Step-1
# connect with database
def connect_database():
    cred = credentials.Certificate('fb_credentials.json')

    # Initialize the app with a service account, granting admin privileges
    firebase_admin.initialize_app(cred, {
        'databaseURL': config_data['databaseURL']})  # getting url stored in config file

    return db


database = connect_database()


#################
"""SignUp Page"""
#################


# Step 2
@app.route("/", methods=['GET', 'POST'])
def signup():
    global capture, error, successfulMsg

    if request.method == 'POST':

        # if Capture Image & SignUp button pressed, save the data into database
        if request.form.get('Signup_action') == 'Capture Image & SignUp':

            email = request.form['email']
            password = request.form['password']
            print(email, password)  # to verify

            # check whether the entered value is empty
            if request.form['email'] == '' or request.form['password'] == '':
                error = 'Please enter values in the input field'

            else:
                capture = 1  # if signup button pressed then set capture=1

                # sending email,password for database operation(saving these values)
                push_data2Database(email, password)

                # Throw msg if user data successfully saved
                successfulMsg = "Data Saved"
                flash(successfulMsg, 'success')  # flash a msg after task is successfully done

                return redirect(url_for('login'))

        # otherwise if login button pressed then switch to login page
        elif request.form.get('login_action') == 'Login':

            return redirect(url_for('login'))

        else:
            pass  # unknown

    return render_template('Signup_page.html', error=error, msg=successfulMsg)


# Generate frames for Live streaming on Signup page
def SignupVideoGen():
    """ Opencv cam """
    # cam = cv2.VideoCapture(2)

    global capture, ImageArray

    # Take user image: [OpenCV]
    while True:

        ret, frame = cam.read()

        if ret:

            if (capture):  # if capture button clicked then capture has set 1 and click img

                capture = 0
                cv2.imwrite('img.jpg', frame)

                # Convert the image into array and store into global np array variable
                ImageArray = np.array(frame)

                # process the image and produce facial embeddings
                convert2FacialEmbeddings()

            # encode the frame so that flask can return it as a response
            ret, buffer = cv2.imencode('.jpg', frame)

            # convert each frame to a byte object
            frame = buffer.tobytes()

            # concat frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# return Frames to the flask signup template
@app.route('/video_feed')
def video_feed():
    return Response(SignupVideoGen(), mimetype='multipart/x-mixed-replace; boundary=frame')



#################
"""Login Page"""
#################


# Step 3
# Task:
# 1. check if current user email exist in db.
# 2. If yes then fetch the appropriate embedded feature for facial verification
@app.route('/login', methods=['POST', 'GET'])
def login():
    global currentUser_embedded_face, dbUser_embedding, error, database

    # refrencing the database
    ref = database.reference('/Users')

    if request.method == 'POST':

        # pressed forget password key
        if request.form.get('forget_action') == 'Forget Password':

            if request.form['email'] == '':
                error = 'Please enter your Email Address'

            # 1.check if email exist in db

            else:

                # get the email id from current user
                email = request.form['email']
                print(email)

                # creating list of child node of db
                child_id = []
                for i in range(1, 5):  # lets us assume we have only 5 users in db
                    child_id.append(str(i))

                # search for child node
                for i in child_id:
                    user = ref.child(i).get()

                    if user != None:

                        if user['email'] == email:  # searching current user in db via email
                            print('found email at child node:', i)

                            # 2. Fetched appropriate Face embidding using emailID as unique identifier
                            encodded_embedding = user['embeddings']
                            print(type(encodded_embedding))

                            # Deserailize the embeddings
                            print("Decode JSON serialized NumPy array")
                            decodded_embedding = json.loads(encodded_embedding)
                            dbUser_embedding = np.array(decodded_embedding["embeddings"])

                            return redirect(url_for('forget_password'))

                        else:
                            error = "Please enter valid email, Signup if new?"

    return render_template('Login_page.html', error=error)


##########################
"""Forget Password Page"""
##########################

# Step 4
# If current User is the existing User in the DB...Then for auth forget password page
"""
Forget Password Page: 
1. Consist of Live streaming Face detection.
2. Button to click the photo for verification
"""


@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    global mcapture
    global euclid_distance, cosine_similarity, threshold_cosine, threshold_euclid

    if request.method == 'POST':

        # if Verify button pressed Start verfication process(MTCNN, Emebedd, Euclid)
        if request.form.get('verify_action') == 'Verify Image':

            print("Verify button clicked")
            mcapture = 1

            # use try except block to release lock because if release function run done twice then error arises. 
            try:
                forgetPage_lock.release()

            except:
                print("Click verify button only once, lock already released'")
                return render_template('ForgetPassword_page.html')

            # for getting the face verification results sleep down this function for 4.5sec
            time.sleep(4.5)

            # recieved verification results, updated by ForgetPass_videoGen function
            print('recieved verification results:')
            print(euclid_distance, cosine_similarity, threshold_cosine, threshold_euclid)

            # If Face verified, give access to the user
            if euclid_distance < threshold_euclid:

                print('Face Verified!!')
                return redirect(url_for('home'))

            else:
                print('Face Not Verified')
                return redirect(url_for('login'))

    return render_template('ForgetPassword_page.html')


mainThread_lock.acquire()  # blocking main thread to avoid ForgetPassVideo run at first call


# Generate frames for streaming on ForgetPassword page
"""
It performs: 
1. Live streaming with Face detection.
2. Facial Verification
"""

def ForgetPassVideoGen():
    global ImageArray, mcapture, x, y, w, h, currentUser_embedding, imgArray
    global euclid_distance, cosine_similarity, threshold_cosine, threshold_euclid

    # after generator run once lock the thread
    forgetPage_lock.acquire()


    # Using Haar cascade for realtime face detection
    # [Motive] -> To display user its detected face which guide them
    #             to click image at correct timing.


    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


    # Take user image: [OpenCV]
    while True:

        ret, frame = cam.read()

        if ret:

            """Displaying Live Face detected bbox for users"""
            # NOTE: cann't use MTCNN for face bbox creation
            #       bcz it is very slow as compared to Haar caascade

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h),
                              color=(255, 0, 0), thickness=3)

            """When verify button is pressed"""
            if (mcapture):  # verification button pressed & mcapture becomes 1

                mcapture = 0

                """Preparing Image for verification Process"""
                # Convert the image into array and store into global numpy array variable
                ImageArray = np.array(frame)

                img = Image.fromarray(ImageArray)  # convert from array to image onject

                img = img.convert(colors='RGB')  # convert image object to RGB

                imgArray = np.array(img)  # converting image RGB back to np array

                """
                Performing Image verification process
                1. Getting CurrentUser emebeddings.
                2. Finall Facial Verification stage(Match the database & currentUser embedding)
                """

                # 1. Current user embedding
                currentUser_embedding = ForgetPassword_page_FaceVerification.currentUser_ImageProcess(imgArray)

                # 2. Facial Verification stage
                euclid_distance, cosine_similarity, threshold_cosine, threshold_euclid = Face_Verifiction_stage03.verify_face(
                    dbUser_embedding, currentUser_embedding)

                """to verify code results"""
                # print('Verification Results:....')
                # print('euclid_distance: ',euclid_distance)
                # print('cosine_similarity: ',cosine_similarity)
                # print('threshold_cosine: ',threshold_cosine)
                # print('threshold_euclid: ',threshold_euclid)

            # encode the frame so that flask can return it as a response in
            ret, buffer = cv2.imencode('.jpg', frame)

            # convert each frame to a byte object
            frame = buffer.tobytes()

            # concat frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Returns video feed to the Forget Password page


@app.route('/ForgetPassVideo_feed')
def ForgetPassVideo_feed():
    return Response(ForgetPassVideoGen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Home
@app.route('/home')
def home():
    return render_template('Home_page.html')


##################################################################################

# For Step-1
# Convert Image to Embedded form [Only for Sigup Clicked Image]
def convert2FacialEmbeddings():
    img = Image.fromarray(ImageArray)  # convert from array to image onject

    img = img.convert(colors='RGB')  # convert image object to RGB

    imgArray = np.array(img)  # converting image RGB back to np array

    # Sends the image array for MTCNN -> Embedded
    # 1. MTCNN
    cropped_face, zipped_coord = ForgetPassword_page_FaceVerification.prepare_mtcnn(imgArray, bool_val=True)

    # 2, Embedded
    embedded_face = ForgetPassword_page_FaceVerification.prepare_Embeddings(cropped_face)

    embedded_face = embedded_face[0, :]  # [0,:] -> to bring the shape from (1,2048) to (2048,)

    return embedded_face


#####################################
# converting np.array in json format
#####################################

# used for converting numpy array emmbedded feature json for pushing in db
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Push user data into database
def push_data2Database(user_email, user_password):
    """ Connecting to database """
    global database

    """ get user data for storing """
    embedded_face = convert2FacialEmbeddings()
    email, password = user_email, user_password

    """ Store values into database """

    """
    We Have three value to store:
    1. Email as string 
    2. Password as String
    3. Embedded face as array, so convert into json and then save to firebase.
    """

    ### Before pushing np.array embedded_face to db convert it to json ##

    # Serialization embedded_face
    numpyData_embedding = {"embeddings": embedded_face}
    encoded_EmbeddedData = json.dumps(numpyData_embedding, cls=NumpyArrayEncoder)  # use dump() to write array into file

    # prepare data to insert
    data = {'email': email, 'password': password, 'embeddings': encoded_EmbeddedData}

    # created reference of database
    ref = database.reference('/Users')

    # Save data into database
    ref.child('1').set(data)


def runFlask():
    app.run()


if __name__ == '__main__':

    thread1 = threading.Thread(target=runFlask, name='thread-1')
    thread2 = threading.Thread(target=ForgetPassVideoGen, name='thread-2')

    thread1.start()
    thread2.start()


    """Small Note"""
    # activate thread2 when we are at ForgetPassword_page
    # else, keep it lock
    # lock.acquire() -> lock thread
    # lock.release() -> unlock thread