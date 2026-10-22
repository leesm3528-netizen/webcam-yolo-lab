"""
<summary>step2_server.py를 확장해, 웹 페이지의 "촬영" 버튼으로 현재 프레임을 jpg로 저장하고 바로 보여 준다.</summary>
<reason>
동영상은 결국 연속된 이미지 한 장 한 장의 모음이다.
스트림에서 한 장을 떼어 파일로 저장해 보면 이 사실을 눈으로 확인할 수 있고,
저장한 jpg는 다음 단계(이미지 한 장에 YOLO 실행)의 실습 재료가 된다.
</reason>
"""

import atexit
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, render_template, send_from_directory

import config


# 서버 주소는 이 실습에서 로컬로 고정한다.
HOST = "127.0.0.1"
PORT = 5000

# 카메라가 연속으로 이 횟수만큼 프레임을 못 읽으면 끊긴 것으로 본다.
MAX_READ_FAILURES = 30

# 캡처 이미지를 저장할 폴더. 실행 위치와 상관없이 이 파일 옆의 captures 폴더를 쓴다.
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

# 촬영 이력으로 보여 줄 최근 사진 수
HISTORY_LIMIT = 10

# 캡처 파일 이름 규칙. 예: 20261022_141530.jpg, 같은 초에 두 번 찍으면 20261022_141530_2.jpg
CAPTURE_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}(_\d+)?\.jpg$")


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
    웹캠은 한 번에 한 곳에서만 열 수 있으므로, 서버가 켜질 때 한 번만 열고 모든 요청이 최신 프레임을 나눠 본다.
    촬영 요청도 카메라를 새로 열지 않고, 스트림이 보고 있는 바로 그 최신 프레임을 가져가 저장한다.
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
        <reason>lock은 읽기 스레드가 프레임을 바꾸는 순간과 웹 요청이 프레임을 가져가는 순간이 겹치지 않게 막는다.</reason>
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
        <reason>Windows에서는 CAP_DSHOW로 열어야 카메라가 빨리 켜지고 경고가 적다.</reason>
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
        <reason>cap.read()는 다음 프레임을 기다리는 함수라, 별도 스레드로 빼야 웹 요청이 막히지 않는다.</reason>
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
        <reason>복사본을 주어야 JPEG 변환이나 파일 저장 도중에 다음 프레임으로 덮어써져도 안전하다.</reason>
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
        <reason>atexit와 오류 처리 양쪽에서 불릴 수 있으므로 두 번째 호출은 아무 일도 하지 않게 한다.</reason>
        """
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[정보] 카메라를 해제했습니다.")


class CaptureStore:
    """
    <summary>캡처 이미지를 captures 폴더에 저장하고, 저장된 목록을 관리한다.</summary>
    <param name="directory">캡처 이미지를 저장할 폴더 경로.</param>
    <input>save()에 BGR 프레임을 넘기고, recent()로 최근 목록을 받는다.</input>
    <output>save()는 저장한 파일 이름, recent()는 파일 이름 목록을 돌려준다.</output>
    <reason>
    저장, 이름 짓기, 목록 조회를 한 클래스로 모으면 라우트 함수는 "요청을 받고 응답한다"는 일에만 집중할 수 있다.
    파일 이름 규칙(CAPTURE_NAME_PATTERN)도 이곳 한 군데에서만 다룬다.
    </reason>
    """

    def __init__(self, directory):
        """
        <summary>저장 폴더를 만들고, 동시 저장을 막을 lock을 준비한다.</summary>
        <param name="directory">캡처 이미지를 저장할 폴더 경로.</param>
        <input>폴더 경로.</input>
        <output>없음.</output>
        <reason>
        버튼을 빠르게 두 번 누르면 두 요청이 동시에 같은 이름을 고를 수 있다.
        이름 고르기와 저장을 lock 안에서 한 번에 처리해 파일이 덮어써지지 않게 한다.
        </reason>
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _next_filename(self):
        """
        <summary>현재 시각으로 YYYYMMDD_HHMMSS.jpg 이름을 만든다. 이미 있으면 _2, _3을 붙인다.</summary>
        <input>없음. 현재 시각을 쓴다.</input>
        <output>아직 쓰이지 않은 파일 이름 문자열.</output>
        <reason>시각을 이름으로 쓰면 파일 이름 순서가 곧 촬영 순서가 되어 정렬이 쉽다.</reason>
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}.jpg"
        number = 2

        while (self.directory / filename).exists():
            filename = f"{stamp}_{number}.jpg"
            number += 1

        return filename

    def save(self, frame):
        """
        <summary>프레임을 jpg 파일로 저장한다.</summary>
        <param name="frame">저장할 BGR 이미지(numpy 배열).</param>
        <input>BGR 이미지.</input>
        <output>저장한 파일 이름 문자열. 저장에 실패하면 None.</output>
        <reason>cv2.imwrite는 실패해도 예외 없이 False만 돌려주므로 결과를 직접 확인한다.</reason>
        """
        with self.lock:
            filename = self._next_filename()
            path = self.directory / filename
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not ok:
            print(f"[오류] 캡처 이미지를 저장하지 못했습니다: {path}")
            return None

        print(f"[정보] 캡처 저장: {path}")
        return filename

    def recent(self, limit):
        """
        <summary>저장된 캡처 파일 이름을 최신순으로 돌려준다.</summary>
        <param name="limit">돌려줄 최대 개수.</param>
        <input>최대 개수.</input>
        <output>파일 이름 문자열 목록. 가장 최근 것이 맨 앞이다.</output>
        <reason>
        이름 규칙에 맞는 파일만 고른다.
        학생이 captures 폴더에 다른 파일을 넣어 두어도 목록에 섞이지 않는다.
        </reason>
        """
        names = [path.name for path in self.directory.glob("*.jpg") if is_valid_capture_name(path.name)]
        names.sort(key=capture_sort_key, reverse=True)

        return names[:limit]


def is_valid_capture_name(filename):
    """
    <summary>파일 이름이 캡처 이름 규칙(YYYYMMDD_HHMMSS.jpg)에 맞는지 확인한다.</summary>
    <param name="filename">확인할 파일 이름.</param>
    <input>파일 이름 문자열.</input>
    <output>규칙에 맞으면 True, 아니면 False.</output>
    <reason>
    "허용할 모양"만 통과시키는 방식(화이트리스트)이다.
    "..", "/", "\" 같은 위험한 문자를 하나하나 막는 것보다 빠뜨릴 틈이 없다.
    </reason>
    """
    return CAPTURE_NAME_PATTERN.match(filename) is not None


def capture_sort_key(filename):
    """
    <summary>캡처 파일 이름을 촬영 순서대로 정렬하기 위한 기준값을 만든다.</summary>
    <param name="filename">규칙에 맞는 캡처 파일 이름.</param>
    <input>파일 이름 문자열.</input>
    <output>(시각 문자열, 같은 초 안의 번호) 튜플.</output>
    <reason>
    글자 순서로만 정렬하면 "_10"이 "_2"보다 앞에 온다.
    같은 초에 찍은 사진의 번호는 숫자로 바꿔 비교해야 순서가 맞다.
    </reason>
    """
    stem = filename[:-4]
    parts = stem.split("_")
    stamp = parts[0] + parts[1]

    if len(parts) == 3:
        number = int(parts[2])
    else:
        number = 1

    return stamp, number


def generate_mjpeg(camera):
    """
    <summary>최신 프레임을 JPEG로 바꿔 MJPEG 형식의 조각으로 계속 내보낸다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <input>Camera 객체.</input>
    <output>MJPEG 조각(bytes)을 하나씩 내보내는 제너레이터.</output>
    <reason>
    MJPEG 스트림은 JPEG 사진을 계속 이어 보내는 것이다.
    촬영 버튼은 이 흐름에서 지금 보내고 있는 사진 한 장을 파일로 남기는 것과 같다.
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


def create_app(camera, store):
    """
    <summary>스트림, 촬영, 캡처 이미지 제공, 촬영 이력 주소를 가진 Flask 앱을 만든다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <param name="store">캡처를 저장하고 관리하는 CaptureStore 객체.</param>
    <input>Camera와 CaptureStore 객체.</input>
    <output>Flask 앱 객체.</output>
    <reason>카메라와 저장소를 인자로 받으면 전역 변수 없이도 라우트에서 쓸 수 있다.</reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>templates/step3_capture.html을 그려서 돌려준다.</summary>
        <input>브라우저의 GET / 요청.</input>
        <output>HTML 응답.</output>
        <reason>2단계의 index.html은 그대로 두고, 3단계용 페이지를 따로 만들어 두 단계를 각각 실행할 수 있게 한다.</reason>
        """
        return render_template("step3_capture.html", history_limit=HISTORY_LIMIT)

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

    @app.route("/capture", methods=["POST"])
    def capture():
        """
        <summary>현재 최신 프레임을 captures 폴더에 jpg로 저장하고 파일 이름을 JSON으로 돌려준다.</summary>
        <input>브라우저의 POST /capture 요청. 본문은 필요 없다.</input>
        <output>성공: {"filename": ..., "url": ...} (200). 실패: {"error": ...} (503 또는 500).</output>
        <reason>
        GET이 아닌 POST를 쓰는 이유는 이 요청이 "서버에 파일을 만드는" 동작이기 때문이다.
        GET은 주소창 미리 불러오기나 새로 고침으로도 다시 보내질 수 있어, 원치 않는 사진이 저장될 수 있다.
        </reason>
        """
        frame_id, frame = camera.get_frame()

        if frame is None or not camera.running:
            return jsonify(error="카메라 영상이 없어 촬영할 수 없습니다. 서버 콘솔을 확인하세요."), 503

        filename = store.save(frame)

        if filename is None:
            return jsonify(error="이미지를 저장하지 못했습니다. captures 폴더 권한을 확인하세요."), 500

        return jsonify(filename=filename, url=f"/captures/{filename}", frame_id=frame_id)

    @app.route("/captures")
    def capture_list():
        """
        <summary>최근 촬영 이력을 JSON으로 돌려준다.</summary>
        <input>브라우저의 GET /captures 요청.</input>
        <output>{"captures": [{"filename": ..., "url": ...}, ...]} 최신순, 최대 HISTORY_LIMIT개.</output>
        <reason>페이지를 새로 고치거나 서버를 다시 켜도, 이미 찍어 둔 사진 이력이 그대로 보이게 한다.</reason>
        """
        items = []

        for name in store.recent(HISTORY_LIMIT):
            items.append({"filename": name, "url": f"/captures/{name}"})

        return jsonify(captures=items)

    @app.route("/captures/<filename>")
    def capture_file(filename):
        """
        <summary>저장된 캡처 이미지 파일을 돌려준다.</summary>
        <param name="filename">주소에 들어온 파일 이름.</param>
        <input>브라우저의 GET /captures/파일이름 요청.</input>
        <output>jpg 이미지 응답. 이름이 규칙에 맞지 않거나 파일이 없으면 404.</output>
        <reason>
        경로 조작(../) 공격을 두 겹으로 막는다.
        첫째, 이름이 캡처 규칙에 맞지 않으면 파일을 찾아보지도 않고 404로 돌려보낸다.
        둘째, send_from_directory는 지정한 폴더 밖의 파일은 보내지 않는다.
        open(CAPTURE_DIR / filename)처럼 직접 경로를 붙이면 ..\\config.py 같은 이름으로 다른 파일이 새어 나갈 수 있다.
        </reason>
        """
        if not is_valid_capture_name(filename):
            abort(404)

        return send_from_directory(store.directory, filename, mimetype="image/jpeg")

    return app


def run():
    """
    <summary>카메라와 캡처 저장소를 준비하고, 종료 시 카메라 해제를 예약한 뒤 Flask 서버를 실행한다.</summary>
    <input>config.py의 카메라 번호, 해상도, JPEG 화질 설정.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    atexit.register는 프로그램이 어떻게 끝나든 마지막에 카메라 해제를 불러 준다.
    debug와 use_reloader를 끄는 이유는 Flask가 프로그램을 한 번 더 실행해 카메라를 두 번 열지 않게 하기 위해서다.
    </reason>
    """
    try:
        store = CaptureStore(CAPTURE_DIR)
    except OSError as error:
        print(f"[오류] 캡처 폴더를 만들지 못했습니다: {error}")
        return 1

    camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if not camera.start():
        print_camera_help(config.CAMERA_INDEX)
        return 1

    atexit.register(camera.release)

    if not camera.wait_first_frame(timeout=5):
        print("[오류] 카메라는 열렸지만 5초 동안 영상이 들어오지 않았습니다.")
        print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
        return 1

    app = create_app(camera, store)

    print(f"[정보] 캡처 저장 폴더: {CAPTURE_DIR}")
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
# 2. 실행 (templates\step3_capture.html이 같은 폴더에 있어야 한다)
#      python step3_capture.py
# 3. 브라우저에서 http://127.0.0.1:5000 에 접속한다.
# 4. "촬영" 버튼을 누르면(또는 스페이스 키) 스트림 옆에 방금 찍은 사진이 나타나고,
#    captures\YYYYMMDD_HHMMSS.jpg 파일이 생긴다.
# 5. 콘솔에서 Ctrl+C를 누르면 서버가 끝나고 카메라가 해제된다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 동영상 = 연속된 이미지 한 장의 모음
#    - 손을 빠르게 흔들면서 촬영해 보자. 저장된 사진은 움직이는가, 멈춰 있는가?
#      스트림은 이런 멈춘 사진을 1초에 수십 장씩 갈아 끼워 움직이는 것처럼 보일 뿐이다.
#    - /capture 응답 JSON의 frame_id 값을 보자. (F12 > 네트워크 > capture)
#      1~2초 간격으로 두 번 찍으면 frame_id는 얼마나 차이 나는가? 그 차이가 그사이 지나간 이미지 수다.
#    - captures 폴더의 jpg를 그림판이나 VS Code로 열어 보자. 그냥 평범한 사진 파일이다.
#
# 2. 저장된 jpg는 다음 단계의 실습 재료
#    - 사람, 컵, 휴대폰, 책 등이 보이게 여러 장 찍어 두자.
#    - 다음 단계에서 이 사진에 YOLO를 실행한다. 예:
#        python step3_yolo_image.py captures\20261022_141530.jpg
#    - 움직이는 영상 대신 멈춘 사진으로 먼저 실험하면, 같은 입력으로 몇 번이고 결과를 비교할 수 있다.
#
# 3. 경로 조작 방지
#    - 브라우저 주소창에 http://127.0.0.1:5000/captures/..%2Fconfig.py 를 넣어 보자. 404가 나와야 정상이다.
#    - capture_file의 reason을 읽고, 왜 이름 규칙 검사를 먼저 하는지 설명해 보자.
