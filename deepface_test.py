# 결과 출력
from deepface import DeepFace

# 이미지 경로
img_path = "assets/people.jpg"

# 얼굴 분석 수행
result = DeepFace.analyze(img_path, actions=('age', 'gender'))
print(len(result))
# 결과 출력
print("Age:", result[0]["age"])
print("Gender:", result[0]["gender"])
