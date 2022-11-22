import numpy as np
import cv2
import os
import hashlib
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, Response, request, redirect, url_for
# from flask_jsglue import JSGlue
from gtts import gTTS
from playsound import playsound
from skimage.transform import resize
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibmcloudant.cloudant_v1 import CloudantV1, Document
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
# from jinja2 import M

app=Flask(__name__)
# jsglue = JSGlue(app)
speechfile=0
audiooutput=0
lock_audio=False
transcript=""
pred_text = "No Gesture was shown"
error=""

def initialize_db():
    global db_name, client
    authenticator = IAMAuthenticator('YIw8TkFJzxfUbZya9rp4YLHBVl3_HHrHk2cnDYjskbkG')
    client = CloudantV1(
        authenticator=authenticator
    )
    client.set_service_url('https://cf4faf1c-2967-4e3a-a69a-209a3c28cc3c-bluemix.cloudantnosqldb.appdomain.cloud')
    db_name = "usersdb"

def initialize_speechToText():
    global speech_to_text
    authenticator = IAMAuthenticator('MGU3du2-gL3o1rTmURlh2UHKkqM8AHnZqtHF1rg5P7pF')
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )
    speech_to_text.set_service_url('https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/24675f56-6aa0-4aaf-abab-e5f8f38af20b')

# def textToSpeech():
#     global speechfile
#     speechfile+=1
#     freq = 44100
#     duration = 10
#     recording = sd.rec(int(duration * freq),samplerate=freq, channels=2)
#     sd.wait()
#     write("audiofiles/audio"+str(speechfile)+".wav", freq, recording)

@app.route('/speech', methods =["GET", "POST"])
def speechToText():
    global speech_to_text, lock_audio, transcript, speechfile
    lock_audio=True
    speechfile+=1
    freq = 44100
    duration = 5
    recording = sd.rec(int(duration * freq),samplerate=freq, channels=2)
    sd.wait()
    write("Application/audiofiles/audio"+str(speechfile)+".wav", freq, recording)
    with open("Application/audiofiles/audio"+str(speechfile)+".wav", 'rb') as audio_file:
        speech_recog=speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            model='en-US_Telephony'
        ).get_result()
    results=speech_recog['results']
    transcript=""
    for i in results:
        transcript+=" "+i['alternatives'][0]['transcript']
    print(transcript)
    lock_audio=False
    return redirect(url_for('index'))

def initialize_model():
    global graph,writer,model,vals,cap,pred,rec, backSub
    rec=False
    backSub = cv2.createBackgroundSubtractorKNN()
    graph=tf.compat.v1.get_default_graph()
    writer=None
    # with graph.as_default():
    model = load_model('Application/weight15.h5')
    vals=['A','B','C','D','E','F','G','H','I']
    print("[INFO] accessing video stream...")
    cap=cv2.VideoCapture(0)
    pred=""

@app.route('/signup', methods =["GET", "POST"])
def sign_up():
    global db_name, client, rev, doc_id
    if request.method == "POST":
        doc_id = request.form.get('mail')
        document: Document = Document(id=doc_id)
        document.fname = request.form.get("fname")
        document.lname = request.form.get("lname")
        document.phone=request.form.get('phone')
        password=request.form.get('password')
        document.password=hashlib.sha256(password.encode()).hexdigest()

        document_response = client.post_document(db=db_name,document=document).get_result()
        document.rev = document_response["rev"]
        rev = document_response["rev"]
    return redirect(url_for('index'))

@app.route('/login', methods =["GET", "POST"])
def login():
    global db_name, client, error, doc_id
    if request.method == "POST":
        doc_id = request.form.get('mail')
        password=request.form.get('password')
        try:
            document = client.get_document(
                db=db_name,
                doc_id=doc_id
            ).get_result()
        except:
            error="Account not Registered. SignUp Here"
            return render_template('login_signup.html', error=error)
        cipher_password=hashlib.sha256(password.encode()).hexdigest()
        if document['password']==cipher_password:
            error=""
            return redirect(url_for('index'))
        else:
            # TODO
            error = "Password is Wrong"
            return render_template('login_signup.html', error=error)

@app.route('/signout', methods =["GET", "POST"])
def signout():
    global db_name, client, error, rev, doc_id
    document = client.delete_document(
        db=db_name,
        doc_id=doc_id,
        rev=rev
    ).get_result()
    return redirect(url_for('login_signup'))

@app.route('/logout', methods =["GET", "POST"])
def logout():
    return redirect(url_for('login_signup'))

def detect(frame):
    global pred_text, audiooutput
    # print("Inside Detect")
    # cv2.imshow('frame', frame)
    mask = backSub.apply(frame)
    # cv2.imshow("Hand Segmentation", mask)
    img = resize(mask, (28,28,1))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img/255
    # with graph.as_default():
    prediction = model.predict(img)
    if len(np.where(prediction[0]>=0.5)[0]) > 0:
        pred=vals[np.where(prediction[0]>=0.5)[0][0]]
        if pred_text == "No Gesture was shown":
            pred_text = ""
            pred_text = pred
        elif pred != pred_text[-1]:
            pred_text = pred_text + " " + pred
        # print("Inside Predict")

# def gen():  
#     while True:
#         success, frame = cap.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen():
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ret1, buffer = cv2.imencode('.png', frame)
            frame1 = buffer.tobytes()
            # if cv2.waitKey(1) == ord('q'):
            #     break
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame1 + b'\r\n')
            detect(frame)
        else:
            # print("Can't receive frame (stream end?). Exiting ...")
            exit
    cap.release()
    cv2.destroyAllWindows()

# @app.route('/js')
# def js():
#     return render_template('script.js')

@app.route('/')
def login_signup():
    # render_template('static/script.js')
    return render_template('login_signup.html', error=error)
    # return render_template('index.html', var=rec, transcript=transcript,lock_audio=lock_audio,pred_text=pred_text)

@app.route('/index')
def index():
    # render_template('static/script.js')
    return render_template('index.html', var=rec,transcript=transcript,lock_audio=lock_audio,pred_text=pred_text)

@app.route('/video')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record', methods =["GET", "POST"])
def record():
    global rec, cap, audiooutput
    if request.method == "POST":
        start_rec = request.form.get('start_rec')
        print(start_rec)
    if rec==False:
        rec=True
    else:
        rec=False
        cap.release()
        myobj = gTTS(text=pred_text, lang='en', slow=False)
        audiooutput+=1
        myobj.save("Application/speechoutput/speech"+str(audiooutput)+".mp3")
        playsound("Application/speechoutput/speech"+str(audiooutput)+".mp3")
    # print(rec)
    return redirect(url_for('index', var=rec))

if __name__=="__main__":
    initialize_db()
    initialize_speechToText()
    initialize_model()
    app.run(host='0.0.0.0')

