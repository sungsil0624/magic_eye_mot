import firebase_admin
from firebase_admin import credentials, db


class RealtimeDBManager:
    def __init__(self, credentials_path, database_url):
        # Firebase Admin SDK 초기화
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})

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

# 예시 사용
# credentials_path = 'path/to/your/firebase/credentials.json'
# database_url = 'https://your-firebase-project-id.firebaseio.com'
# firebase_manager = FirebaseManager(credentials_path, database_url)
#
# # 예시: 데이터 쓰기
# firebase_manager.write_data('example_path', {'key1': 'value1', 'key2': 'value2'})
#
# # 예시: 데이터 조회
# data = firebase_manager.read_data('example_path')
# print(data)
#
# # 예시: 데이터 삭제
# firebase_manager.delete_data('example_path')
