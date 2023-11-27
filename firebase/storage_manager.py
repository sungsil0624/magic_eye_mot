import firebase_admin
from firebase_admin import credentials, storage


class StorageManager:
    def __init__(self, credentials_path, bucket_name):
        # Firebase Admin SDK 초기화
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {'storageBucket': f'{bucket_name}.appspot.com'})

    def upload_video(self, local_file_path, remote_file_path):
        # 영상 업로드
        bucket = storage.bucket()
        blob = bucket.blob(remote_file_path)
        blob.upload_from_filename(local_file_path)

    def upload_image(self, local_file_path, remote_file_path):
        # 사진 업로드
        bucket = storage.bucket()
        blob = bucket.blob(remote_file_path)
        blob.upload_from_filename(local_file_path)