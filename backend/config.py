
# Our Firebase configuration for backend operations.
import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Load the environment variables.
load_dotenv()

# Init. Firebase Admin SDK.
if "FIREBASE_ADMIN_SDK" not in os.environ:
    raise ValueError("FIREBASE_ADMIN_SDK environment variable is not set.")

# Load the Firebase Admin SDK credentials from the environment variable.
cred = credentials.Certificate(json.loads(os.environ["FIREBASE_ADMIN_SDK"]))
firebase_admin.initialize_app(cred, {
    'storageBucket': "nst-flask-app.firebasestorage.app"
})