import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS
import pdfplumber
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Video generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    prev_center = None
    movement_threshold = 20
    missing_frame_count = 0
    max_missing_frames = 10
    head_move_count = 0
    head_moved = False
    eye_lost = False
    face_missing = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            missing_frame_count = 0
            face_missing = False

            for (x, y, w, h) in faces:
                current_center = (x + w // 2, y + h // 2)

                # Head movement detection
                if prev_center is not None:
                    dist = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                    if dist > movement_threshold:
                        cv2.putText(frame, "WARNING: Head movement!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    

                prev_center = current_center

                

                # Eye detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                eye_contact_lost = False

                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                    _, thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        center = int(cx)
                        eye_width = eye_roi.shape[1]

                        if center < eye_width * 0.3 or center > eye_width * 0.7:
                            eye_contact_lost = True

                if eye_contact_lost:
                    cv2.putText(frame,  (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                else:
                    eye_lost = False

                break  # one face only
        else:
            missing_frame_count += 1
            if missing_frame_count > max_missing_frames:
                cv2.putText(frame, "WARNING: Face not visible!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not face_missing:
                    print("Face not visible")
                    face_missing = True
        if head_move_count > 4:
            break

        # Encode frame to JPEG for web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

conversation_step = {}
user_data = {}

def bot_reply(user_id, user_message=None):
    step = conversation_step.get(user_id, 0)

    if step == 0:
        reply = "Hello! I am your chatbot. 👋 Please upload your resume (PDF)."
        conversation_step[user_id] = 1

    elif step == 1:
        reply = "Waiting for your resume upload..."
    
    elif step == 2:
        lang = user_data.get(user_id, {}).get("language")
        reply = f"I see you're interested in {lang}. Are you an experienced professional or a fresher?"
        conversation_step[user_id] = 3

    elif step == 3:
        if "experience" in user_message.lower():
            reply = "Awesome! So Lets Go " 
            "1"
        elif "fresher" in user_message.lower():
            reply = "Fantastic! SO Lets GO !"
        else:
            reply = "Please tell me if you are experienced or a fresher."
        conversation_step[user_id] = 4

    elif step == 4:
            lang = user_data.get(user_id, {}).get("language")
            reply = f" Let me Ask some Question about  {lang} are you ready! "
            conversation_step[user_id] = 5
   
    elif step == 5:
        reply = "What is Python? What are the benefits of using Python"
        conversation_step[user_id] = 6
    
    elif step == 6 :
          reply = "That's amazing god job! Now next Question"
          reply = "What is the difference between a mutable data type and an immutable data type?"
          conversation_step[user_id] = 7

    elif step == 7 :
           reply = "Great Job ! Have a nice day! 👍 "
           
    else:
        reply = "Thank you for chatting with me! You can refresh the page to start again."

    return reply


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['resume']
    user_id = "user1"
    conversation_step[user_id] = 2

    if file and file.filename.endswith(".pdf"):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Detect language based on PDF content
        if "python" in text.lower():
            language = "Python"
        elif "java" in text.lower():
            language = "Java"
        else:
            language = "an unknown language"

        user_data[user_id] = {"language": language}

        reply = bot_reply(user_id)
        return jsonify({"reply": reply})
    else:
        return jsonify({"reply": "Please upload a valid PDF file."})
    


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = "user1"
    user_message = data.get("message", "")
    reply = bot_reply(user_id, user_message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
