import os
from typing import Any, Dict
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

# _db = None

def _get_db(credentials_path: str) -> firestore.Client:
    if credentials_path is None:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        global _db
    if not firebase_admin._apps:  # check if already initialized
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

    # cred = credentials.Certificate(credentials_path)
    # app = initialize_app(cred)
    # return firestore.client(app)

def save_document(collection: str, data: Dict[str, Any], credentials_path: str) -> None:
    """Save a document to a Firestore collection."""
    db = _get_db(credentials_path)
    db.collection(collection).add(data)
