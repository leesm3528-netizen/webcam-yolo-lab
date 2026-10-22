"""
<summary>Flask 로컬 서버(127.0.0.1:5000)를 띄워 웹캠 영상을 MJPEG 스트림(/video_feed)으로 브라우저에 보낸다.</summary>
<reason>
cv2.imshow 창 대신 브라우저로 영상을 보면, 이후 YOLO 결과도 같은 방식으로 웹에 얹을 수 있다.
카메라 읽기는 Camera 클래스로 분리해, 웹 요청 처리와 서로 얽히지 않게 한다.
</reason>
"""

import atexit
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template

import config


# 서버 주소는 이 실습에서 로컬로 고정한다.
HOST = "127.0.0.1"
PORT = 5000

# 카메라가 연속으로 이 횟수만큼 프레임을 못 읽으면 끊긴 것으로 본다.
MAX_READ_FAILURES = 30


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


class Camera:
    """
    <summary>웹캠을 한 번만 열고, 별도 스레드에서 계속 읽어 최신 프레임 1장만 보관한다.</summary>
    <param name="index">카메라 번호. 내장 카메라는 보통 0이다.</param>
    <param name="width">요청할 가로 해상도(픽셀).</param>
    <param name="height">요청할 세로 해상도(픽셀).</param>
    <input>start()로 읽기를 시작하고, get_frame()으로 최신 프레임을 가져간다.</input>
    <output>get_frame()이 (프레임 번호, BGR 이미지) 튜플을 돌려준다.</output>
    <reason>
    요청마다 카메라를 열면 장치 충돌이 나기 때문이다.
    웹캠은 한 번에 한 곳에서만 열 수 있어서, 브라우저 탭이 두 개만 되어도 두 번째 요청은 카메라를 열지 못한다.
    그래서 카메라는 서버가 켜질 때 한 번만 열고, 읽기 스레드 하나가 계속 읽으며, 모든 요청은 최신 프레임을 나눠 본다.
    최신 1장만 남기는 이유는 오래된 프레임이 쌓이면 화면이 실제보다 늦게 보이기 때문이다.
    </reason>
    """

    def __init__(self, index, width, height):
        """
        <summary>카메라 설정과 스레드에서 쓸 값을 준비한다. 카메라는 아직 열지 않는다.</summary>
        <param name="index">카메라 번호.</param>
        <param name="width">요청할 가로 해상도.</param>
        <param name="height">요청할 세로 해상도.</param>
        <input>카메라 번호와 해상도.</input>
        <output>없음.</output>
        <reason>
        lock은 읽기 스레드가 프레임을 바꾸는 순간과 웹 요청 스레드가 프레임을 가져가는 순간이 겹치지 않게 막는다.
        겹치면 반쯤 바뀐 프레임을 보내거나, 프레임과 프레임 번호가 서로 어긋날 수 있다.
        </reason>
        """
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.thread = None

    def start(self):
        """
        <summary>카메라를 열고 프레임 읽기 스레드를 시작한다.</summary>
        <input>없음.</input>
        <output>성공하면 True, 카메라를 열지 못하면 False.</output>
        <reason>
        Windows에서는 CAP_DSHOW로 열어야 카메라가 빨리 켜지고 경고가 적다.
        daemon 스레드로 만들면 메인 프로그램이 끝날 때 스레드가 남아 프로그램을 붙잡지 않는다.
        </reason>
        """
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        return True

    def _read_loop(self):
        """
        <summary>스레드 안에서 카메라 프레임을 계속 읽어 최신 프레임을 덮어쓴다.</summary>
        <input>없음.</input>
        <output>없음. self.frame과 self.frame_id가 바뀐다.</output>
        <reason>
        cap.read()는 다음 프레임이 올 때까지 기다리는 함수라, 웹 요청 안에서 부르면 응답이 그만큼 막힌다.
        읽기를 별도 스레드로 빼면 웹 요청은 기다리지 않고 이미 읽어 둔 프레임을 바로 가져간다.
        한 번 실패로 멈추지 않고, 연속으로 실패했을 때만 카메라가 끊긴 것으로 본다.
        </reason>
        """
        fail_count = 0

        while self.running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                fail_count += 1

                if fail_count >= MAX_READ_FAILURES:
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

    def get_frame(self):
        """
        <summary>최신 프레임의 복사본과 프레임 번호를 돌려준다.</summary>
        <input>없음.</input>
        <output>(프레임 번호, BGR 이미지) 튜플. 아직 프레임이 없으면 (0, None).</output>
        <reason>복사본을 주어야 요청 쪽에서 JPEG로 바꾸는 동안 읽기 스레드가 다음 프레임으로 덮어써도 안전하다.</reason>
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
            frame_id, _ = self.get_frame()

            if frame_id > 0:
                return True

            if not self.running:
                return False

            time.sleep(0.05)

        return False

    def release(self):
        """
        <summary>읽기 스레드를 멈추고 카메라를 해제한다. 여러 번 불러도 안전하다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>
        atexit와 오류 처리 양쪽에서 불릴 수 있으므로 두 번째 호출은 아무 일도 하지 않게 한다.
        스레드가 cap.read() 중일 때 release()하면 오류가 날 수 있어, 스레드가 끝나길 기다린 뒤 해제한다.
        </reason>
        """
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[정보] 카메라를 해제했습니다.")


def generate_mjpeg(camera):
    """
    <summary>최신 프레임을 JPEG로 바꿔 MJPEG 형식의 조각으로 계속 내보낸다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <input>Camera 객체.</input>
    <output>MJPEG 조각(bytes)을 하나씩 내보내는 제너레이터.</output>
    <reason>
    MJPEG는 JPEG 사진 여러 장을 경계 문자열(--frame)로 이어 붙여 한 응답으로 계속 보내는 방식이다.
    조각 하나는 "--frame, 헤더(Content-Type), 빈 줄, JPEG 데이터" 순서로 되어 있다.
    같은 프레임 번호면 다시 보내지 않아 네트워크와 CPU를 아낀다.
    </reason>
    """
    last_id = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]

    while camera.running:
        frame_id, frame = camera.get_frame()

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
    <summary>index 페이지와 /video_feed 스트림 주소를 가진 Flask 앱을 만든다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <input>Camera 객체.</input>
    <output>Flask 앱 객체.</output>
    <reason>
    HTML을 templates/index.html로 분리하면 화면 디자인과 서버 코드를 따로 고칠 수 있다.
    카메라 객체를 인자로 받으면 전역 변수 없이도 라우트에서 카메라를 쓸 수 있다.
    </reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>templates/index.html을 그려서 돌려준다.</summary>
        <input>브라우저의 GET / 요청.</input>
        <output>HTML 응답.</output>
        <reason>render_template은 이 파일과 같은 위치의 templates 폴더에서 HTML을 찾는다.</reason>
        """
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>MJPEG 영상 스트림을 돌려준다.</summary>
        <input>브라우저의 GET /video_feed 요청.</input>
        <output>multipart/x-mixed-replace 형식의 스트리밍 응답.</output>
        <reason>
        multipart/x-mixed-replace는 "다음 조각이 오면 앞 조각을 갈아 끼워라"라는 뜻이다.
        boundary=frame은 조각 사이의 경계 문자열이 --frame이라고 브라우저에 알려 준다.
        이 응답은 끝나지 않고 계속 이어지므로, 브라우저의 img 태그가 동영상처럼 보여 준다.
        </reason>
        """
        return Response(
            generate_mjpeg(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def run():
    """
    <summary>카메라를 열고 종료 시 해제를 예약한 뒤 Flask 서버를 실행한다.</summary>
    <input>config.py의 카메라 번호, 해상도, JPEG 화질 설정.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    atexit.register는 프로그램이 어떻게 끝나든(Ctrl+C, 정상 종료, 오류) 마지막에 함수를 불러 준다.
    카메라를 열자마자 등록해 두면 해제를 잊을 일이 없다.
    debug와 use_reloader를 끄는 이유는 Flask가 프로그램을 한 번 더 실행해 카메라를 두 번 열지 않게 하기 위해서다.
    threaded=True는 영상 스트림이 계속 이어지는 동안에도 다른 요청을 받을 수 있게 한다.
    </reason>
    """
    camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if not camera.start():
        print_camera_help(config.CAMERA_INDEX)
        return 1

    atexit.register(camera.release)

    if not camera.wait_first_frame(timeout=5):
        print("[오류] 카메라는 열렸지만 5초 동안 영상이 들어오지 않았습니다.")
        print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
        return 1

    app = create_app(camera)

    print(f"[정보] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print("[정보] 서버를 끄려면 이 콘솔에서 Ctrl+C를 누르세요.")

    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    except KeyboardInterrupt:
        print("[정보] Ctrl+C로 서버를 끝냅니다.")

    except OSError as error:
        print(f"[오류] 서버를 시작하지 못했습니다: {error}")
        print(f"{PORT}번 포트를 다른 프로그램이 쓰고 있지 않은지 확인하세요.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행 (templates\index.html이 같은 폴더에 있어야 한다)
#      python step2_server.py
# 3. 브라우저에서 http://127.0.0.1:5000 에 접속한다.
# 4. 콘솔에서 Ctrl+C를 누르면 서버가 끝나고, atexit가 카메라를 해제한다.
#    콘솔에 "[정보] 카메라를 해제했습니다."가 나오는지 확인한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. multipart/x-mixed-replace 응답 구조
#    - PowerShell에서 아래 명령으로 스트림 앞부분을 직접 받아 보자. (2초 후 끊는다)
#        curl.exe -s -m 2 http://127.0.0.1:5000/video_feed -o stream.bin
#    - VS Code에서 stream.bin을 열면 "--frame", "Content-Type: image/jpeg", 빈 줄,
#      알아볼 수 없는 JPEG 데이터가 반복되는 것이 보인다. 이 반복 하나가 프레임 한 장이다.
#    - 브라우저 개발자 도구(F12) > 네트워크 탭에서 video_feed 요청의 응답 헤더 Content-Type을 확인하자.
#      이 요청은 "완료"되지 않고 계속 받는 중으로 남는다. 왜 그럴까?
#
# 2. 스레드와 lock이 필요한 이유
#    - 브라우저 탭을 세 개 열어도 모든 탭에서 영상이 나오는가?
#      카메라를 요청마다 열었다면 두 번째 탭부터 장치 충돌로 영상이 안 나왔을 것이다.
#    - Camera._read_loop가 없고 video_feed 안에서 cap.read()를 직접 불렀다면,
#      탭 세 개가 프레임을 나눠 가져가서 각 탭의 FPS는 어떻게 될까?
#    - get_frame()에서 "with self.lock:"을 지우면 당장은 잘 돌아가 보일 수 있다.
#      그런데도 lock이 필요한 이유를 Camera.__init__의 reason을 읽고 설명해 보자.
