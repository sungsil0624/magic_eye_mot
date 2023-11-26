from datetime import datetime


# 현재 시각 리턴
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
