from typing import Any, Dict
import firebase_admin
from firebase_admin import credentials, firestore

_db = None

def _get_db(credentials_path: str) -> firestore.Client:
    global _db
    if _db is None:
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db

def save_document(collection: str, data: Dict[str, Any], credentials_path: str) -> None:
    """Save a document to a Firestore collection."""
    db = _get_db(credentials_path)
    db.collection(collection).add(data)
