from datetime import datetime


# 현재 시각 리턴
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 0초인지 확인
def is_exact_minute():
    current_time = datetime.now()
    return current_time.second == 0
