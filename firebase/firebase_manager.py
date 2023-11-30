import firebase_admin
from firebase_admin import credentials, storage, db


class FireBaseManager:
    def __init__(self, credentials_path, database_url, bucket_name):
        # Firebase Admin SDK 초기화
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url,
                                             'storageBucket': f'{bucket_name}.appspot.com'})

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

    def write_data(self, path, data):
        # 데이터 쓰기
        ref = db.reference(path)
        ref.set(data)

    def update_data(self, path, updates):
        # 데이터 업데이트
        ref = db.reference(path)
        ref.update(updates)

    def delete_data(self, path):
        # 데이터 삭제
        ref = db.reference(path)
        ref.delete()

    def read_data(self, path):
        # 데이터 조회
        ref = db.reference(path)
        return ref.get()
