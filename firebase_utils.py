import os
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st


def _initialize_firebase_app(cred: credentials.Base) -> None:
    """Firebase Admin SDKを初期化する。既に初期化済みの場合は何もしない。"""

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)


def _get_credentials_from_streamlit() -> Optional[credentials.Certificate]:
    """Streamlit secretsからサービスアカウント資格情報を取得する。"""

    try:
        sa_info = dict(st.secrets["gcp_service_account"])
    except (AttributeError, KeyError, RuntimeError):
        return None
    except Exception:
        # streamlit側で予期せぬ例外が出た場合は、他の認証手段にフォールバックする
        return None

    private_key = sa_info.get("private_key")
    if isinstance(private_key, str) and "\\n" in private_key:
        sa_info["private_key"] = private_key.replace("\\n", "\n")

    return credentials.Certificate(sa_info)


def _get_default_credentials() -> credentials.Base:
    """利用可能な認証情報から優先順位に沿って資格情報を取得する。"""

    streamlit_credentials = _get_credentials_from_streamlit()
    if streamlit_credentials is not None:
        return streamlit_credentials

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_path}"
            )
        return credentials.Certificate(credentials_path)

    try:
        return credentials.ApplicationDefault()
    except Exception as exc:
        raise RuntimeError(
            "No valid Firestore credentials found. Please configure Streamlit "
            "secrets, set GOOGLE_APPLICATION_CREDENTIALS, or provide a "
            "credentials file."
        ) from exc


def _get_db_from_secrets() -> firestore.Client:
    """Streamlit secretsからGCPサービスアカウント情報を取得してFirestore接続"""

    cred = _get_default_credentials()
    _initialize_firebase_app(cred)
    return firestore.client()


def _get_db_from_credentials_file(credentials_path: str) -> firestore.Client:
    """ファイルパスで指定されたサービスアカウント情報からFirestoreへ接続"""

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

    cred = credentials.Certificate(credentials_path)
    _initialize_firebase_app(cred)
    return firestore.client()


def save_document(
    collection: str, data: Dict[str, Any], credentials_path: Optional[str] = None
) -> None:
    """Firestoreコレクションにドキュメントを保存"""

    if credentials_path:
        db = _get_db_from_credentials_file(credentials_path)
    else:
        db = _get_db_from_secrets()
    db.collection(collection).add(data)
