import firebase_admin
from firebase_admin import credentials, messaging


class PushNotificationManager:
    def __init__(self, credentials_path):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)

    def send_push_notification(self, registration_token, title, body, data=None):
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data=data,
            token=registration_token,
        )

        try:
            response = messaging.send(message)
            print("Successfully sent message:", response)
        except Exception as e:
            print("Error sending message:", e)


# 예시 사용
# credentials_path = 'path/to/your/firebase/credentials.json'
# push_notification_manager = PushNotificationManager(credentials_path)
# registration_token = 'device_registration_token'
# title = 'Notification Title'
# body = 'Notification Body'
# push_notification_manager.send_push_notification(registration_token, title, body)
