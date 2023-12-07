import vlc
import time


def play_vlc_stream(url):
    # VLC 인스턴스 생성
    instance = vlc.Instance('--no-xlib')

    # 미디어 플레이어 생성
    player = instance.media_player_new()

    # 스트리밍 URL 설정
    media = instance.media_new(url)
    media.get_mrl()

    # 플레이어에 미디어 설정
    player.set_media(media)

    # 미디어 플레이어 실행
    player.play()

    # 잠시 대기 (예: 30초 동안 재생)
    time.sleep(30)

    # 플레이어 정지
    player.stop()
