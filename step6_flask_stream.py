"""
<summary>6단계: Flask 웹 서버로 웹캠 영상을 브라우저에 실시간 전송한다(MJPEG 스트리밍).</summary>
<reason>
cv2.imshow 창은 그 컴퓨터 앞에서만 볼 수 있다.
웹으로 내보내면 휴대폰이나 다른 컴퓨터의 브라우저에서도 볼 수 있고, 7단계에서 YOLO 결과를 그대로 얹을 수 있다.
YOLO 없이 스트리밍만 먼저 해 보는 이유는, 문제가 생겼을 때 원인이 웹 쪽인지 모델 쪽인지 나누어 보기 위해서다.
</reason>
"""

import sys
import threading
import time

import cv2
from flask import Flask, Response

import config


INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>6단계 웹캠 스트리밍</title>
  <style>
    body { margin: 0; padding: 16px; background: #111; color: #eee; font-family: sans-serif; }
    img { max-width: 100%; border: 1px solid #444; }
  </style>
</head>
<body>
  <h1>6단계: 웹캠 스트리밍</h1>
  <img src="/video_feed" alt="웹캠 영상">
  <p>영상이 멈추면 서버 콘솔에 출력된 오류 메시지를 확인하세요.</p>
</body>
</html>
"""


def open_camera(index, width, height):
    """
    <summary>지정한 번호의 웹캠을 열고 해상도를 설정한다.</summary>
    <param name="index">카메라 번호. 내장 카메라는 보통 0이다.</param>
    <param name="width">요청할 가로 해상도(픽셀).</param>
    <param name="height">요청할 세로 해상도(픽셀).</param>
    <input>카메라 번호와 해상도.</input>
    <output>열린 cv2.VideoCapture 객체. 열지 못하면 None.</output>
    <reason>Windows에서는 CAP_DSHOW로 열어야 카메라가 빨리 켜지고 경고가 적다.</reason>
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def print_camera_help(index):
    """
    <summary>카메라를 열지 못했을 때 확인할 항목을 출력한다.</summary>
    <param name="index">열기에 실패한 카메라 번호.</param>
    <input>카메라 번호.</input>
    <output>없음. 콘솔에 안내 문구를 출력한다.</output>
    <reason>자주 겪는 원인을 알려 주면 학생이 스스로 문제를 해결할 수 있다.</reason>
    """
    print(f"[오류] {index}번 카메라를 열 수 없습니다.")
    print("다음을 확인하세요.")
    print("  1. 줌, 팀즈, 카메라 앱 등 다른 프로그램이 카메라를 쓰고 있지 않은가?")
    print("  2. Windows 설정 > 개인 정보 및 보안 > 카메라 에서 '데스크톱 앱의 카메라 액세스'가 켜져 있는가?")
    print("  3. 노트북의 카메라 차단 스위치나 Fn 단축키로 카메라가 꺼져 있지 않은가?")
    print("  4. 외장 카메라라면 config.py의 CAMERA_INDEX를 1, 2로 바꿔 보았는가?")


class CameraStream:
    """
    <summary>별도 스레드에서 웹캠을 계속 읽고, 가장 최근 프레임 한 장을 보관한다.</summary>
    <param name="index">카메라 번호.</param>
    <param name="width">요청할 가로 해상도.</param>
    <param name="height">요청할 세로 해상도.</param>
    <input>start()로 읽기를 시작하고, read()로 최근 프레임을 가져간다.</input>
    <output>read()가 (프레임 번호, BGR 이미지) 튜플을 돌려준다.</output>
    <reason>
    웹캠은 하나인데 브라우저 접속은 여러 개일 수 있다.
    접속마다 cap.read()를 부르면 서로 프레임을 빼앗아 영상이 끊기므로, 읽기는 한 스레드만 맡고 접속들은 최근 프레임을 나눠 본다.
    </reason>
    """

    def __init__(self, index, width, height):
        """
        <summary>카메라를 열고 스레드에서 쓸 값을 준비한다.</summary>
        <param name="index">카메라 번호.</param>
        <param name="width">요청할 가로 해상도.</param>
        <param name="height">요청할 세로 해상도.</param>
        <input>카메라 번호와 해상도.</input>
        <output>없음. 카메라를 열지 못하면 self.cap이 None이 된다.</output>
        <reason>Lock은 읽기 스레드와 웹 요청 스레드가 같은 프레임을 동시에 건드리지 않게 막는다.</reason>
        """
        self.index = index
        self.cap = open_camera(index, width, height)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.thread = None

    def start(self):
        """
        <summary>프레임 읽기 스레드를 시작한다.</summary>
        <input>없음.</input>
        <output>시작에 성공하면 True, 카메라가 열려 있지 않으면 False.</output>
        <reason>daemon 스레드로 만들면 메인 프로그램이 끝날 때 함께 정리된다.</reason>
        """
        if self.cap is None:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        return True

    def _loop(self):
        """
        <summary>스레드 안에서 카메라 프레임을 계속 읽어 최근 프레임을 바꿔 둔다.</summary>
        <input>없음.</input>
        <output>없음. self.frame과 self.frame_id가 바뀐다.</output>
        <reason>
        프레임 한 번 실패로 바로 멈추면 일시적인 끊김에도 서버가 죽는다.
        연속으로 30번 실패했을 때만 카메라가 끊긴 것으로 보고 멈춘다.
        </reason>
        """
        fail_count = 0

        while self.running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                fail_count += 1

                if fail_count >= 30:
                    print("[오류] 카메라에서 프레임을 계속 읽지 못했습니다.")
                    print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                    self.running = False
                    break

                time.sleep(0.05)
                continue

            fail_count = 0

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        """
        <summary>가장 최근 프레임의 복사본과 프레임 번호를 돌려준다.</summary>
        <input>없음.</input>
        <output>(프레임 번호, BGR 이미지) 튜플. 아직 프레임이 없으면 (0, None).</output>
        <reason>복사본을 주어야 다른 스레드가 다음 프레임으로 바꾸는 중에도 안전하게 쓸 수 있다.</reason>
        """
        with self.lock:
            if self.frame is None:
                return 0, None

            return self.frame_id, self.frame.copy()

    def wait_first_frame(self, timeout):
        """
        <summary>첫 프레임이 들어올 때까지 기다린다.</summary>
        <param name="timeout">최대 대기 시간(초).</param>
        <input>대기 시간.</input>
        <output>시간 안에 프레임이 들어오면 True, 아니면 False.</output>
        <reason>카메라가 열리긴 했지만 영상이 안 나오는 경우를 서버 시작 전에 걸러낸다.</reason>
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            frame_id, _ = self.read()

            if frame_id > 0:
                return True

            if not self.running:
                return False

            time.sleep(0.05)

        return False

    def stop(self):
        """
        <summary>읽기 스레드를 멈추고 카메라를 해제한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>스레드가 cap.read()를 쓰는 중에 release()하면 오류가 날 수 있어, 스레드가 끝나길 기다린 뒤 해제한다.</reason>
        """
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()


def generate_mjpeg(camera):
    """
    <summary>최근 프레임을 JPEG로 바꿔 MJPEG 형식으로 끝없이 내보낸다.</summary>
    <param name="camera">실행 중인 CameraStream 객체.</param>
    <input>CameraStream 객체.</input>
    <output>MJPEG 조각(bytes)을 하나씩 내보내는 제너레이터.</output>
    <reason>
    MJPEG는 JPEG 사진을 경계 문자열(--frame)로 이어 붙여 계속 보내는 방식이다.
    브라우저가 &lt;img&gt; 태그만으로 재생할 수 있어 별도 플레이어가 필요 없다.
    같은 프레임 번호면 다시 보내지 않아 네트워크와 CPU를 아낀다.
    </reason>
    """
    last_id = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]

    while camera.running:
        frame_id, frame = camera.read()

        if frame is None or frame_id == last_id:
            time.sleep(0.01)
            continue

        last_id = frame_id

        ok, buffer = cv2.imencode(".jpg", frame, encode_params)

        if not ok:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def create_app(camera):
    """
    <summary>웹 페이지와 영상 스트림 주소를 가진 Flask 앱을 만든다.</summary>
    <param name="camera">실행 중인 CameraStream 객체.</param>
    <input>CameraStream 객체.</input>
    <output>Flask 앱 객체.</output>
    <reason>카메라 객체를 인자로 받아 앱을 만들면 전역 변수 없이도 라우트에서 카메라를 쓸 수 있다.</reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>영상이 들어갈 &lt;img&gt; 태그가 있는 HTML 페이지를 돌려준다.</summary>
        <input>브라우저의 GET / 요청.</input>
        <output>HTML 문자열.</output>
        <reason>페이지와 영상 주소를 나누면 페이지 디자인을 바꿔도 스트림 코드는 그대로 둘 수 있다.</reason>
        """
        return INDEX_HTML

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>MJPEG 영상 스트림을 돌려준다.</summary>
        <input>브라우저의 GET /video_feed 요청.</input>
        <output>multipart/x-mixed-replace 형식의 스트리밍 응답.</output>
        <reason>이 mimetype이어야 브라우저가 새 JPEG가 올 때마다 이전 그림을 바꿔 끼운다.</reason>
        """
        return Response(
            generate_mjpeg(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def run():
    """
    <summary>카메라 스레드를 시작하고 Flask 서버를 실행한다.</summary>
    <input>config.py의 카메라, Flask 주소와 포트, JPEG 화질 설정.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 카메라 오류는 1.</output>
    <reason>
    debug=True로 켜면 Flask가 프로그램을 한 번 더 실행해 카메라를 두 번 열려고 한다.
    그래서 debug와 자동 재시작(use_reloader)을 끈다.
    </reason>
    """
    camera = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if not camera.start():
        print_camera_help(config.CAMERA_INDEX)
        return 1

    try:
        if not camera.wait_first_frame(timeout=5):
            print("[오류] 카메라는 열렸지만 5초 동안 영상이 들어오지 않았습니다.")
            print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
            return 1

        app = create_app(camera)

        print(f"[정보] 브라우저에서 http://{config.FLASK_HOST}:{config.FLASK_PORT} 에 접속하세요.")
        print("[정보] 서버를 끄려면 이 콘솔에서 Ctrl+C를 누르세요.")

        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    except KeyboardInterrupt:
        print("[정보] Ctrl+C로 서버를 끝냅니다.")

    except OSError as error:
        print(f"[오류] 서버를 시작하지 못했습니다: {error}")
        print(f"{config.FLASK_PORT}번 포트를 다른 프로그램이 쓰고 있다면 config.py의 FLASK_PORT를 바꾸세요.")
        return 1

    finally:
        camera.stop()
        print("[정보] 카메라를 해제했습니다.")

    return 0


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행
#      python step6_flask_stream.py
# 3. 브라우저에서 http://127.0.0.1:5000 에 접속한다.
# 4. 콘솔에서 Ctrl+C를 누르면 서버와 카메라가 함께 종료된다.
#
# 같은 와이파이의 휴대폰에서 보려면
#   - config.py의 FLASK_HOST를 "0.0.0.0"으로 바꾼다.
#   - PowerShell에서 ipconfig로 내 IPv4 주소(예: 192.168.0.12)를 확인한다.
#   - 휴대폰 브라우저에서 http://192.168.0.12:5000 에 접속한다.
#   - Windows 방화벽 허용 창이 뜨면 "개인 네트워크"를 허용한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - 브라우저 탭을 두세 개 열어도 영상이 모두 잘 나오는가? (CameraStream의 reason)
# - 브라우저 개발자 도구(F12) > 네트워크 탭에서 video_feed 요청은 끝나지 않고 계속 이어지는가?
# - config.py의 JPEG_QUALITY를 30과 95로 바꾸면 화질과 끊김이 어떻게 달라지는가?
# - 서버를 끈 뒤 브라우저 화면은 어떻게 되는가?
