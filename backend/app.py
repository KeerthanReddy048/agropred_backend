import smtplib
import numpy as np
import joblib
from flask import Flask, session, request, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from pymongo import MongoClient
from datetime import timedelta
from flask import Flask, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from dotenv import load_dotenv
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
app = Flask(__name__)
bcrypt = Bcrypt(app)
load_dotenv()
# CORS Configuration
CORS(app, supports_credentials=True)
SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
# Secret key for sessions
app.secret_key = ' '
# Set session lifetime and make it permanent
app.permanent_session_lifetime = timedelta(days=7)

# MongoDB Configuration
client = MongoClient("mongodb+srv://dbUser:GVRReddy123@cluster0.fw9zc.mongodb.net/agropredDB?retryWrites=true&w=majority")
db = client['user_auth']  # Database name
users_collection = db['users']  # Collection name

@app.before_request
def make_session_permanent():
    session.permanent = True


models = {}
traits = ['DH_Pooled', 'GFD_Pooled', 'GNPS_Pooled', 'GWPS_Pooled', 'PH_Pooled', 'GY_Pooled']

for trait in traits:
    with open(f"C:/Users/keert/OneDrive/Desktop/AgroPredA-19/flask-backend/{trait}_model.pkl", "rb") as f:
        models[trait] = joblib.load(f)

# Load scalers
with open("C:/Users/keert/OneDrive/Desktop/AgroPredA-19/flask-backend/scaler_X.pkl", "rb") as f:
    loaded_scaler_X = joblib.load(f)

with open("C:/Users/keert/OneDrive/Desktop/AgroPredA-19/flask-backend/scaler_y.pkl", "rb") as f:
    loaded_scaler_y = joblib.load(f)


"""import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys"""

# Paths


@app.route('/predict', methods=['POST'])
def get_pred():
    try:
        data = request.json
        input_data = np.array(data['ip']).reshape(1, -1)

        # Transform the input data using the trained scaler
        scaled_input_data = loaded_scaler_X.transform(input_data)

        # Make predictions for all traits
        predictions_scaled = [models[trait].predict(scaled_input_data)[0] for trait in traits]

        # Inverse transform to original scale
        predictions = loaded_scaler_y.inverse_transform(np.array(predictions_scaled).reshape(1, -1))

        # Format predictions
        result = {trait: round(float(predictions[0][i]), 3) for i, trait in enumerate(traits)}

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
#model = joblib.load(r"C:\Users\keert\OneDrive\Desktop\AgroPredA-19\flask-backend\random_forest_model.pkl")
#loaded_scaler_X = joblib.load(r'C:\Users\keert\OneDrive\Desktop\AgroPredA-19\flask-backend\scaler_X.pkl')
#loaded_scaler_y = joblib.load(r'C:\Users\keert\OneDrive\Desktop\AgroPredA-19\flask-backend\scaler_y.pkl')


#from flask import Flask, request, jsonify
#from flask_cors import CORS
"""import torch
import torchvision
from torchvision import transforms
from PIL import Image
import io

#app = Flask(__name__)
#CORS(app)  # Enable CORS for cross-origin requests

# Load Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(r"C:\Users\keert\OneDrive\Desktop\global-wheat-detection\fasterrcnn_resnet50_epochUpdated.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocess function
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)
count=0
# Extract bounding boxes
def get_boxes(prediction, threshold=0.5):
    boxes = []
    count=0
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(score)})
            count+=1
    if(count>=15):
        return boxes
    else:
        boxes=[]
        return boxes

model_path = r"C:\Users\keert\Downloads\image_source_classifier.h5"  # Path to the trained model


# Parameters
IMG_SIZE = (128, 128)

# Load the model
model_2 = load_model(model_path)

#label_map = {0: 'arvalis_1', 1: 'arvalis_2', 2: 'arvalis_3', 3: 'ethz_1', 4: 'inrae_1', 5: 'rres_1', 6: 'usask_1'}
label_map = {0: 'Post-flowering', 1: 'Filling', 2: 'Filling - Ripening', 3: 'Filling', 4: 'Filling - Ripening', 5: 'Filling - Ripening', 6: 'Filling - Ripening'}

@app.route("/spike-detection", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    input_tensor = preprocess_image(image.convert("RGB"))
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction_2=model_2(img_array)[0]
        predicted_label = label_map[np.argmax(prediction_2)]
    boxes = get_boxes(prediction)
    result = {
        "boxes": boxes,
        "development_stage": predicted_label
    }
    if(len(boxes)==0):
        result="Image Error"
    return jsonify(result)
    #return jsonify("Image Error")"""

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"     # Gmail SMTP server
SMTP_PORT = 587
EMAIL_ADDRESS = "gkreddy200389@gmail.com"        # Replace with your Gmail
EMAIL_PASSWORD = "lcfn ebem lftu ncmg"           # Replace with your Gmail app password
SENDER_NAME = "AgroPred"                         # Custom sender name

# Function to send "Thank You" email
def send_thank_you_email(email, username):
    subject = "Thank You for Using Our Platform!"
    
    # HTML email body
    body = f"""
    <html>
    <head></head>
    <body>
        <p>Hi {username},</p>
        <p>Thank you for registering on <b>AgroPred</b>!</p>
        <p>We’re excited to have you with us. Start exploring and making the most of our platform.</p>
        <p>Regards,<br>AgroPred Team</p>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart()
        
        # Setting email headers
        msg['From'] = f"{SENDER_NAME} <{EMAIL_ADDRESS}>"
        msg['To'] = email
        msg['Subject'] = subject

        # Attach HTML content
        msg.attach(MIMEText(body, 'html'))

        # SMTP connection
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("Thank You email sent successfully!")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# 🔥 Generate verification token
def generate_token(email):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def send_login_email(email,username):
    subject = "Thank You for Using Our Platform!"
    
    # HTML email body
    body = f"""
    <html>
    <head></head>
    <body>
        <p>Hi {username},</p>
        <p>A New Login Found on <b>AgroPred</b>!</p>
        <p>Regards,<br>AgroPred Team</p>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart()
        
        # Setting email headers
        msg['From'] = f"{SENDER_NAME} <{EMAIL_ADDRESS}>"
        msg['To'] = email
        msg['Subject'] = subject

        # Attach HTML content
        msg.attach(MIMEText(body, 'html'))

        # SMTP connection
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("Login email sent successfully!")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# 📧 Send verification email
def send_verification_email(email, username):
    try:
        token = generate_token(email)
        verify_link = f"http://127.0.0.1:5000/api/verify/{token}"

        # Create message container
        msg = MIMEMultipart("alternative")
        msg['From'] = f"{SENDER_NAME} <{EMAIL_ADDRESS}>"
        msg['To'] = email
        msg['Subject'] = "Verify Your Email"
        # HTML content
        html_content = f"""
        <html>
        <body>
            <h2>Hello, {username}!</h2>
            <p>Thank you for signing up. Please click the button below to verify your email:</p>
            <p>
                <a href="{verify_link}" style="background-color: #4CAF50; color: white; padding: 12px 20px; text-decoration: none; display: inline-block; border-radius: 5px;">Verify Email</a>
            </p>
            <p>If you didn't request this, please ignore this email.</p>
            <hr>
            <footer>
                <p>AgroPred Team</p>
            </footer>
        </body>
        </html>
        """

        # Attach HTML content
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        with SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, email, msg.as_string())

        print(f"Verification email sent to {email}")
        return True

    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


# 🚀 Registration Route
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not (username and email and password):
        return jsonify({"message": "All fields are required."}), 400

    # Check if the email is already registered
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already registered."}), 409

    hashed_password = generate_password_hash(password)

    # Store the user in MongoDB as unverified
    user = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "is_verified": False
    }
    users_collection.insert_one(user)

    # Send verification email
    if send_verification_email(email, username) and send_thank_you_email(email,username):
        return jsonify({"message": "Registration successful. Verification email sent!"}), 201
    else:
        return jsonify({"message": "Registration successful, but failed to send verification email."}), 500


# ✅ Email verification route
@app.route('/api/verify/<token>', methods=['GET'])
def verify_email(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get('email')

        # Verify user and update the database
        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"message": "Invalid or expired token."}), 400

        if user["is_verified"]:
            return jsonify({"message": "Email already verified."}), 200

        # Update verification status
        users_collection.update_one({"email": email}, {"$set": {"is_verified": True}})
        return jsonify({"message": "Email verified successfully!"}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token expired. Please register again."}), 400
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid token."}), 400


# 🛠️ Login route with email verification check
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    username = users_collection.find_one({"email": email})
    if not (email and password):
        return jsonify({"message": "All fields are required."}), 400

    # Check if user exists
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"message": "Invalid email or password."}), 401

    # Check if the email is verified
    if not user["is_verified"]:
        return jsonify({"message": "Please verify your email before logging in."}), 403

    # Generate JWT token
    token = jwt.encode({"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, SECRET_KEY, algorithm="HS256")
    send_login_email(email,username['username'])
    return jsonify({"message": "Login successful!", "token": token}), 200



if __name__ == '__main__':
    app.run(debug=True)

