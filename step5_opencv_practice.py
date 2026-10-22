"""
<summary>5단계: 웹캠 실시간 영상에 키보드로 영상 처리 모드(원본, 그레이, 블러, 엣지, 이진화, HSV 마스크)를 바꿔 적용한다.</summary>
<reason>
STEP 4에서 사진 한 장으로 익힌 OpenCV 함수를 실시간 영상에 적용한다.
모드마다 연산량이 달라서, 처리 시간과 FPS가 어떻게 변하는지 직접 체감할 수 있다.
이 감각은 무거운 YOLO를 실시간으로 돌릴 때 "왜 느린가, 어떻게 빠르게 할까"를 이해하는 바탕이 된다.
</reason>
"""

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config


BASE_DIR = Path(__file__).resolve().parent

# s 키로 저장한 결과가 들어갈 폴더
OUTPUT_DIR = BASE_DIR / "outputs"

# 메인 창과 HSV 조절 창의 이름
MAIN_WINDOW = "Step5 - OpenCV Practice"
HSV_WINDOW = "Step5 - HSV Range"

# FPS와 처리 시간을 평균 낼 최근 프레임 수
AVERAGE_WINDOW = 30

# 가우시안 블러 커널 크기(홀수). 클수록 더 흐려지고 더 느려진다.
BLUR_KERNEL = 21

# Canny 엣지의 낮은/높은 임계값
CANNY_LOW = 50
CANNY_HIGH = 150

# 이진화 기준값. 이 값보다 밝으면 흰색(255), 어두우면 검정(0)
THRESHOLD_VALUE = 127

# HSV 트랙바 기본값. 파란색 계열 물체를 잡는 범위로 시작한다.
HSV_DEFAULTS = {
    "H min": 90,
    "H max": 130,
    "S min": 80,
    "S max": 255,
    "V min": 50,
    "V max": 255,
}

# 트랙바마다 움직일 수 있는 최대값. OpenCV의 H는 0~179, S와 V는 0~255다.
HSV_LIMITS = {
    "H min": 179,
    "H max": 179,
    "S min": 255,
    "S max": 255,
    "V min": 255,
    "V max": 255,
}

# 저장 완료 메시지를 화면에 보여 줄 시간(초)
SAVE_MESSAGE_SECONDS = 1.5


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


class MovingAverageMeter:
    """
    <summary>최근 N프레임의 시각과 처리 시간을 모아 FPS와 평균 처리 시간을 계산한다.</summary>
    <param name="size">평균 낼 최근 프레임 수.</param>
    <input>프레임마다 tick(처리 시간 ms)을 호출한다.</input>
    <output>fps(), process_ms()가 이동평균 값을 돌려준다.</output>
    <reason>
    프레임 하나만 보고 계산하면 숫자가 심하게 흔들린다.
    최근 N프레임만 담는 deque(maxlen=N)를 쓰면, 오래된 값은 자동으로 빠지고 평균이 부드럽게 따라온다.
    시각은 time.perf_counter()로 잰다. 짧은 시간 간격을 재기에 가장 정밀한 시계다.
    </reason>
    """

    def __init__(self, size):
        """
        <summary>프레임 시각과 처리 시간을 담을 상자를 만든다.</summary>
        <param name="size">평균 낼 최근 프레임 수.</param>
        <input>정수.</input>
        <output>없음.</output>
        <reason>maxlen을 정하면 상자가 가득 찼을 때 가장 오래된 값이 저절로 밀려난다.</reason>
        """
        self.times = deque(maxlen=size)
        self.process_times = deque(maxlen=size)

    def tick(self, process_ms):
        """
        <summary>프레임 하나가 끝났음을 기록한다.</summary>
        <param name="process_ms">이 프레임의 영상 처리에 걸린 시간(밀리초).</param>
        <input>처리 시간.</input>
        <output>없음.</output>
        <reason>화면 FPS와 처리 시간을 따로 모아야, 카메라 속도 제한과 연산량을 구분해서 볼 수 있다.</reason>
        """
        self.times.append(time.perf_counter())
        self.process_times.append(process_ms)

    def fps(self):
        """
        <summary>최근 프레임들의 평균 FPS를 계산한다.</summary>
        <input>없음.</input>
        <output>FPS(float). 프레임이 2개 미만이면 0.0.</output>
        <reason>(프레임 수 - 1) / (마지막 시각 - 첫 시각)은 최근 구간에서 1초에 몇 장을 처리했는지를 뜻한다.</reason>
        """
        if len(self.times) < 2:
            return 0.0

        elapsed = self.times[-1] - self.times[0]

        if elapsed <= 0:
            return 0.0

        return (len(self.times) - 1) / elapsed

    def process_ms(self):
        """
        <summary>최근 프레임들의 평균 처리 시간을 계산한다.</summary>
        <input>없음.</input>
        <output>평균 처리 시간(밀리초, float). 기록이 없으면 0.0.</output>
        <reason>카메라가 FPS를 30 근처로 묶어 두면 가벼운 모드끼리는 FPS 차이가 안 보이므로, 처리 시간을 함께 본다.</reason>
        """
        if not self.process_times:
            return 0.0

        return sum(self.process_times) / len(self.process_times)

    def reset(self):
        """
        <summary>모아 둔 기록을 모두 지운다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>모드를 바꿀 때 지워야, 새 모드의 숫자에 이전 모드의 값이 섞이지 않는다.</reason>
        """
        self.times.clear()
        self.process_times.clear()


class HsvControls:
    """
    <summary>HSV 색 범위를 조절하는 트랙바 창을 열고, 현재 범위를 읽는다.</summary>
    <param name="window_name">트랙바를 붙일 창 이름.</param>
    <input>open()으로 창을 열고 read()로 값을 읽는다.</input>
    <output>read()가 (lower, upper) numpy 배열 한 쌍을 돌려준다.</output>
    <reason>
    트랙바를 메인 창에 붙이면 영상 창 크기가 바뀌고 가려진다.
    별도 창으로 분리해 HSV 모드일 때만 열면 다른 모드 화면이 깔끔하다.
    창을 닫았다 다시 열어도 마지막 값을 기억하도록 값을 따로 보관한다.
    </reason>
    """

    def __init__(self, window_name):
        """
        <summary>트랙바 값을 기본값으로 준비한다. 창은 아직 열지 않는다.</summary>
        <param name="window_name">트랙바 창 이름.</param>
        <input>창 이름.</input>
        <output>없음.</output>
        <reason>값을 딕셔너리로 들고 있으면 창을 다시 만들 때 마지막 값으로 되살릴 수 있다.</reason>
        """
        self.window_name = window_name
        self.values = dict(HSV_DEFAULTS)
        self.is_open = False

    def open(self):
        """
        <summary>트랙바 창을 만들고 H, S, V의 최소/최대 트랙바를 붙인다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>createTrackbar는 값이 바뀔 때 부를 함수를 요구하지만, 여기서는 매 프레임 read()로 읽으므로 빈 함수를 넘긴다.</reason>
        """
        if self.is_open:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 420, 300)

        for name, value in self.values.items():
            cv2.createTrackbar(name, self.window_name, value, HSV_LIMITS[name], lambda _: None)

        self.is_open = True

    def close(self):
        """
        <summary>트랙바 창을 닫는다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>HSV 모드가 아닐 때는 창을 치워 화면을 정리한다.</reason>
        """
        if not self.is_open:
            return

        cv2.destroyWindow(self.window_name)
        self.is_open = False

    def read(self):
        """
        <summary>트랙바에서 현재 H, S, V 범위를 읽는다.</summary>
        <input>없음.</input>
        <output>(lower, upper) 튜플. 각각 [H, S, V] 순서의 uint8 numpy 배열.</output>
        <reason>
        학생이 X 버튼으로 트랙바 창을 닫아 버리면 getTrackbarPos가 오류를 낸다.
        창이 사라졌으면 마지막 값으로 다시 열어 실습이 끊기지 않게 한다.
        </reason>
        """
        if not self.is_open or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.is_open = False
            self.open()

        for name in self.values:
            self.values[name] = cv2.getTrackbarPos(name, self.window_name)

        lower = np.array([self.values["H min"], self.values["S min"], self.values["V min"]], dtype=np.uint8)
        upper = np.array([self.values["H max"], self.values["S max"], self.values["V max"]], dtype=np.uint8)

        return lower, upper


def to_bgr(image):
    """
    <summary>1채널(흑백) 결과를 3채널로 바꾼다. 이미 3채널이면 그대로 돌려준다.</summary>
    <param name="image">처리 결과 이미지.</param>
    <input>2차원 또는 3차원 배열.</input>
    <output>세로 x 가로 x 3 BGR 이미지.</output>
    <reason>1채널 이미지에는 컬러 글자와 상자를 그릴 수 없으므로, 화면 정보를 그리기 전에 3채널로 맞춘다.</reason>
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def process_original(frame, controls):
    """
    <summary>모드 1: 원본을 그대로 돌려준다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">HsvControls 객체(이 모드에서는 쓰지 않는다).</param>
    <input>BGR 프레임.</input>
    <output>(결과 BGR 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>다른 모드의 처리 시간을 비교할 기준점(연산 0)이 된다.</reason>
    """
    return frame, []


def process_gray(frame, controls):
    """
    <summary>모드 2: 그레이스케일로 바꾼다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">HsvControls 객체(쓰지 않는다).</param>
    <input>BGR 프레임.</input>
    <output>(결과 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>픽셀마다 곱셈 3번과 덧셈 2번뿐이라 가장 가벼운 연산 중 하나다.</reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray, []


def process_blur(frame, controls):
    """
    <summary>모드 3: 가우시안 블러로 영상을 흐리게 한다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">HsvControls 객체(쓰지 않는다).</param>
    <input>BGR 프레임.</input>
    <output>(결과 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>
    픽셀 하나마다 주변 BLUR_KERNEL x BLUR_KERNEL 칸의 가중 평균을 구하므로, 커널이 클수록 연산이 크게 늘어난다.
    잡음을 줄이는 효과가 있어 엣지 검출이나 이진화 전에 자주 쓴다.
    </reason>
    """
    blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)

    return blurred, [f"Kernel: {BLUR_KERNEL}x{BLUR_KERNEL}"]


def process_canny(frame, controls):
    """
    <summary>모드 4: Canny 알고리즘으로 윤곽선(엣지)을 찾는다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">HsvControls 객체(쓰지 않는다).</param>
    <input>BGR 프레임.</input>
    <output>(결과 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>
    그레이 변환, 작은 블러, 기울기 계산, 가는 선 남기기, 두 임계값으로 연결하기까지 여러 단계를 거친다.
    밝기가 급하게 바뀌는 곳이 물체의 경계라는 생각은 CNN의 첫 층이 배우는 특징과도 닮았다.
    </reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    return edges, [f"Low/High: {CANNY_LOW}/{CANNY_HIGH}"]


def process_threshold(frame, controls):
    """
    <summary>모드 5: 밝기 기준으로 흰색과 검정 두 가지로만 나눈다(이진화).</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">HsvControls 객체(쓰지 않는다).</param>
    <input>BGR 프레임.</input>
    <output>(결과 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>픽셀마다 비교 한 번이면 끝나는 가벼운 연산이지만, 조명이 바뀌면 결과가 크게 흔들린다는 한계도 보여 준다.</reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    white_ratio = cv2.countNonZero(binary) / binary.size * 100

    return binary, [f"Threshold: {THRESHOLD_VALUE}", f"White: {white_ratio:.0f}%"]


def process_hsv_mask(frame, controls):
    """
    <summary>모드 6: 트랙바로 고른 HSV 범위의 색만 남기고 나머지는 검게 지운다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="controls">트랙바 값을 읽을 HsvControls 객체.</param>
    <input>BGR 프레임과 트랙바 값.</input>
    <output>(결과 이미지, 추가 설명 문자열 목록) 튜플.</output>
    <reason>
    BGR은 밝기와 색이 섞여 있어 "파란색"을 숫자 범위로 정하기 어렵다.
    HSV는 색상(H), 채도(S), 명도(V)가 나뉘어 있어, 조명이 조금 바뀌어도 H 범위로 같은 색을 잡기 쉽다.
    빨간색처럼 H가 179를 넘어 0으로 돌아가는 색은 H min이 H max보다 크게 두면 두 구간을 합쳐서 잡는다.
    </reason>
    """
    lower, upper = controls.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        low_part = cv2.inRange(hsv, np.array([0, lower[1], lower[2]], dtype=np.uint8), upper)
        high_part = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]], dtype=np.uint8))
        mask = cv2.bitwise_or(low_part, high_part)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    ratio = cv2.countNonZero(mask) / mask.size * 100

    return result, [
        f"H {lower[0]}-{upper[0]}  S {lower[1]}-{upper[1]}  V {lower[2]}-{upper[2]}",
        f"Masked: {ratio:.1f}%",
    ]


# 키 → (화면에 보일 모드 이름, 처리 함수)
MODES = {
    ord("1"): ("1 Original", process_original),
    ord("2"): ("2 Gray", process_gray),
    ord("3"): ("3 Gaussian Blur", process_blur),
    ord("4"): ("4 Canny Edge", process_canny),
    ord("5"): ("5 Threshold", process_threshold),
    ord("6"): ("6 HSV Mask", process_hsv_mask),
}

HSV_MODE_KEY = ord("6")


def draw_info_panel(image, lines, origin=(10, 10)):
    """
    <summary>cv2.rectangle로 반투명 배경 상자를 깔고, 그 위에 cv2.putText로 여러 줄의 글자를 쓴다.</summary>
    <param name="image">글자를 그릴 BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="lines">(문자열, BGR 색) 튜플 목록. 한 원소가 한 줄이다.</param>
    <param name="origin">상자 왼쪽 위 좌표 (x, y).</param>
    <input>BGR 이미지와 줄 목록.</input>
    <output>없음. image가 직접 바뀐다.</output>
    <reason>
    cv2.rectangle과 cv2.putText는 STEP 6~7에서 YOLO 검출 상자와 "person 0.87" 같은 라벨을 그릴 때 그대로 쓰인다.
    여기서 좌표 (x, y) 순서, getTextSize로 글자 크기를 재서 배경 상자를 맞추는 방법, 두께 -1(채우기)을 미리 익혀 둔다.
    cv2.putText는 한글을 못 그리므로 화면 글자는 영어로 쓴다.
    </reason>
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 26
    padding = 10

    x, y = origin
    box_width = 0

    for text, _ in lines:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        box_width = max(box_width, text_width)

    box_right = x + box_width + padding * 2
    box_bottom = y + line_height * len(lines) + padding

    # 반투명 배경: 복사본에 꽉 찬 상자(두께 -1)를 그린 뒤 원본과 섞는다.
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (box_right, box_bottom), (0, 0, 0), -1)
    # 이진화처럼 흰 화면에서도 글자가 읽히도록 배경을 75% 어둡게 덮는다.
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, dst=image)

    # 테두리: 두께 1의 빈 상자. YOLO 검출 상자도 이렇게 그린다.
    cv2.rectangle(image, (x, y), (box_right, box_bottom), (200, 200, 200), 1)

    for i, (text, color) in enumerate(lines):
        text_y = y + padding + line_height * (i + 1) - 8
        cv2.putText(image, text, (x + padding, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_help_bar(image):
    """
    <summary>화면 아래쪽에 키 조작 안내를 한 줄로 그린다.</summary>
    <param name="image">BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <input>BGR 이미지.</input>
    <output>없음.</output>
    <reason>학생이 콘솔을 보지 않아도 화면만 보고 조작할 수 있게 한다.</reason>
    """
    height, width = image.shape[:2]
    text = "1-6: mode   s: save   q: quit"

    cv2.rectangle(image, (0, height - 28), (width, height), (0, 0, 0), -1)
    cv2.putText(image, text, (10, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def save_result(image, mode_name):
    """
    <summary>현재 처리 결과를 outputs 폴더에 jpg로 저장한다.</summary>
    <param name="image">저장할 처리 결과 이미지(화면 글자가 없는 상태).</param>
    <param name="mode_name">현재 모드 이름. 파일 이름에 들어간다.</param>
    <input>이미지와 모드 이름.</input>
    <output>저장한 파일의 Path. 실패하면 None.</output>
    <reason>
    FPS 글자가 얹히기 전의 처리 결과만 저장해야, 다음 단계에서 실습 재료로 쓰거나 보고서에 넣기 좋다.
    파일 이름에 모드와 시각을 넣어 여러 장을 저장해도 덮어쓰지 않는다.
    </reason>
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = mode_name.split(" ", 1)[1].lower().replace(" ", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = OUTPUT_DIR / f"step5_{slug}_{stamp}.jpg"

    if not cv2.imwrite(str(path), image):
        print(f"[오류] 결과를 저장하지 못했습니다: {path}")
        return None

    print(f"[정보] 저장했습니다: {path}")
    return path


def run():
    """
    <summary>웹캠 프레임마다 현재 모드의 처리를 적용하고, 모드 이름과 FPS를 표시한다.</summary>
    <input>config.py의 카메라 설정. 키보드 입력(1~6, s, q).</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 카메라 오류는 1.</output>
    <reason>
    처리 시간은 처리 함수 하나만 감싸서 잰다.
    카메라 읽기와 화면 표시까지 포함하면 연산량 차이가 다른 시간에 묻혀 잘 보이지 않는다.
    </reason>
    """
    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if cap is None:
        print_camera_help(config.CAMERA_INDEX)
        return 1

    controls = HsvControls(HSV_WINDOW)
    meter = MovingAverageMeter(AVERAGE_WINDOW)
    mode_key = ord("1")
    save_message_until = 0.0

    print("[정보] 카메라가 열렸습니다.")
    print("[조작] 1: 원본  2: 그레이  3: 블러  4: 엣지  5: 이진화  6: HSV 마스크  s: 저장  q: 종료")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
                print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            mode_name, process = MODES[mode_key]

            start = time.perf_counter()
            result, extra_lines = process(frame, controls)
            process_ms = (time.perf_counter() - start) * 1000

            meter.tick(process_ms)

            # 저장용 결과(result)와 화면용 이미지(display)를 나눈다.
            display = to_bgr(result).copy()

            average_ms = meter.process_ms()

            # 처리만 놓고 보면 1초에 몇 장까지 가능한가. 카메라 속도 제한에 가려진 연산량 차이를 드러낸다.
            if average_ms > 0:
                capacity = f"{1000 / average_ms:.0f} fps"
            else:
                capacity = "-"

            lines = [
                (f"Mode: {mode_name}", (255, 255, 255)),
                (f"FPS: {meter.fps():.1f}", (120, 230, 120)),
                (f"Process: {average_ms:.2f} ms", (120, 230, 120)),
                (f"Capacity: {capacity}", (120, 230, 120)),
            ]

            for text in extra_lines:
                lines.append((text, (200, 200, 200)))

            if time.perf_counter() < save_message_until:
                lines.append(("Saved!", (80, 200, 255)))

            draw_info_panel(display, lines)
            draw_help_bar(display)

            cv2.imshow(MAIN_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("[정보] 종료 키를 눌러 프로그램을 끝냅니다.")
                break

            if key in MODES and key != mode_key:
                mode_key = key
                meter.reset()
                print(f"[정보] 모드 변경: {MODES[mode_key][0]}")

                if mode_key == HSV_MODE_KEY:
                    controls.open()
                else:
                    controls.close()

            if key == ord("s"):
                if save_result(result, mode_name) is not None:
                    save_message_until = time.perf_counter() + SAVE_MESSAGE_SECONDS

            if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("[정보] 창이 닫혀 프로그램을 끝냅니다.")
                break

    except KeyboardInterrupt:
        print("[정보] Ctrl+C로 프로그램을 끝냅니다.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[정보] 카메라와 창을 해제했습니다.")

    return 0


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행
#      python step5_opencv_practice.py
# 3. 영상 창을 클릭한 뒤 키를 누른다.
#      1: 원본   2: 그레이   3: 가우시안 블러   4: Canny 엣지   5: 이진화   6: HSV 색상 마스크
#      s: 현재 처리 결과 저장 (outputs\step5_모드_시각.jpg)
#      q: 종료
# 4. 6번 모드에서는 "Step5 - HSV Range" 창이 따로 열린다. 트랙바로 H/S/V 범위를 조절한다.
#    트랙바를 움직인 뒤에는 다시 영상 창을 클릭해야 키 입력이 들어간다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 모드별 FPS와 처리 시간을 표로 적어 보자.
#      모드          | FPS | Process(ms) | Capacity
#      1 Original    |     |             |
#      2 Gray        |     |             |
#      3 Blur        |     |             |
#      4 Canny       |     |             |
#      5 Threshold   |     |             |
#      6 HSV Mask    |     |             |
#    - FPS     : 실제로 화면이 1초에 몇 번 바뀌었는가
#    - Process : 영상 처리 함수 하나에 걸린 시간
#    - Capacity: 처리만 놓고 보면 1초에 몇 장까지 가능한가 (= 1000 / Process)
#
# 2. 연산량이 실시간성에 미치는 영향
#    - 모든 모드에서 FPS가 거의 같은데 Process와 Capacity는 크게 다르다. 왜 그럴까?
#      (카메라가 1초에 보내 주는 장수, 보통 15~30장이 상한이다.
#       처리가 그보다 빠르면 FPS는 카메라 속도에 묶이고, 연산량 차이는 Capacity에만 드러난다)
#    - 원본과 HSV 마스크의 Capacity는 몇 배 차이 나는가? 그 차이가 곧 연산량 차이다.
#    - 3번 모드에서 BLUR_KERNEL을 21 → 51로 키우면 Process 값이 어떻게 변하는가?
#      커널 한 변이 약 2.4배가 되면 칸 수(가로 x 세로)는 약 5.9배다. 실제로 잰 값도 5.9배인가?
#      (힌트: 가우시안 블러는 가로 방향, 세로 방향으로 나눠서 두 번 계산할 수 있다.
#       그러면 연산량은 칸 수보다 한 변의 길이에 가깝게 늘어난다. 알고리즘이 연산량을 줄이는 예다)
#    - 대략 "FPS ≈ 1000 / (카메라 대기 + 처리 시간)"이다.
#      처리 시간이 카메라 한 프레임 시간(30FPS면 33ms, 15FPS면 67ms)을 넘으면 FPS가 떨어지기 시작한다.
#      Capacity가 카메라 FPS보다 낮아지는 순간이 바로 그때다.
#    - 이 표를 보관해 두었다가 STEP 6에서 잴 YOLO 추론 시간(ms)과 비교해 보자. 블러보다 몇 배 무거운가?
#      그것이 CPU에서 YOLO를 실시간으로 돌리기 어려운 이유다.
#
# 3. HSV 마스크
#    - 파란색 물건을 비추고 H 범위를 좁혀 그 물건만 남겨 보자.
#    - 빨간색을 잡으려면 H min=170, H max=10처럼 min을 max보다 크게 둔다. 왜 그래야 할까?
#    - 형광등을 껐다 켜거나 물건을 그늘로 옮기면 마스크가 어떻게 변하는가?
#      색 범위만으로 물체를 찾는 방식의 한계가 YOLO 같은 학습 모델이 필요한 이유다.
#
# 4. 그리기 함수
#    - draw_info_panel에서 cv2.rectangle의 두께 -1과 1은 각각 무엇을 그리는가?
#    - STEP 6~7에서 YOLO 상자를 그릴 때 이 함수의 어떤 부분을 그대로 쓰게 될지 예상해 보자.
