import os
from dotenv import load_dotenv
from firebase.firebase_manager import FireBaseManager

load_dotenv()

credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
database_url = os.getenv("FIREBASE_DATABASE_URL")
bucket_name = os.getenv("FIRESTORE_BUCKET_NAME")

storage_manager = FireBaseManager(credentials_path, database_url, bucket_name)

image_local_path = "../assets/people.jpg"
image_remote_path = "images/image.jpg"

storage_manager.upload_image(image_local_path, image_remote_path)
print("Image uploaded successfully.")

data_to_write = {"name": "John", "age": 25, "city": "New York"}
storage_manager.write_data("users/user1", data_to_write)
print("Data written successfully.")
