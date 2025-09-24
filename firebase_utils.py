import os
from typing import Any, Dict, Optional
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

def _get_db_from_secrets() -> firestore.Client:
    """Streamlit secretsからGCPサービスアカウント情報を取得してFirestore接続"""
    sa_info = dict(st.secrets["gcp_service_account"])
    # private_keyの改行コードが正しく渡っているか確認
    if "\\n" in sa_info["private_key"]:
        sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_info)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _get_db_from_credentials_file(credentials_path: str) -> firestore.Client:
    """ファイルパスで指定されたサービスアカウント情報からFirestoreへ接続"""
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_document(collection: str, data: Dict[str, Any], credentials_path: Optional[str] = None) -> None:
    """Firestoreコレクションにドキュメントを保存"""
    if credentials_path:
        db = _get_db_from_credentials_file(credentials_path)
    else:
        db = _get_db_from_secrets()
    db.collection(collection).add(data)
