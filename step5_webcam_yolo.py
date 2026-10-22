"""
<summary>5단계: 웹캠 프레임마다 YOLO를 실행해 실시간으로 물체를 인식하고 창에 표시한다.</summary>
<reason>
1~2단계의 웹캠 반복문과 3~4단계의 YOLO 추론, 상자 그리기를 하나로 합친다.
이미지 한 장에서 동작하던 코드가 연속된 프레임에서도 그대로 동작한다는 것을 확인한다.
</reason>
"""

import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config


PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (211, 188, 0),
    (209, 133, 0),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
]


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


class FpsCounter:
    """
    <summary>프레임 사이 시간 간격으로 초당 프레임 수(FPS)를 계산한다.</summary>
    <param name="smoothing">이전 FPS를 얼마나 유지할지 정하는 값(0~1).</param>
    <input>매 프레임마다 update()를 호출한다.</input>
    <output>update()가 부드럽게 다듬은 FPS 값을 돌려준다.</output>
    <reason>프레임마다 값이 크게 흔들리므로 지수 이동 평균으로 다듬어 읽기 쉽게 한다.</reason>
    """

    def __init__(self, smoothing=0.9):
        """
        <summary>FPS 계산에 필요한 값을 초기화한다.</summary>
        <param name="smoothing">지수 이동 평균의 가중치(0~1).</param>
        <input>가중치 값.</input>
        <output>없음.</output>
        <reason>time.perf_counter()는 짧은 시간 간격을 재기에 가장 정확한 시계다.</reason>
        """
        self.smoothing = smoothing
        self.fps = 0.0
        self.last_time = time.perf_counter()

    def update(self):
        """
        <summary>직전 호출 이후 지난 시간으로 FPS를 새로 계산한다.</summary>
        <input>없음.</input>
        <output>다듬어진 FPS 값(float).</output>
        <reason>첫 값은 그대로 쓰고 이후에만 평균을 내야 처음에 0에서 천천히 올라가지 않는다.</reason>
        """
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now

        if elapsed <= 0:
            return self.fps

        current = 1.0 / elapsed

        if self.fps == 0.0:
            self.fps = current
        else:
            self.fps = self.smoothing * self.fps + (1.0 - self.smoothing) * current

        return self.fps


def load_model(model_path, device):
    """
    <summary>YOLO 모델을 불러오고, 빈 이미지로 한 번 미리 실행해 둔다.</summary>
    <param name="model_path">모델 파일 이름 또는 경로. 예: "yolo11n.pt"</param>
    <param name="device">추론에 쓸 장치 이름. 예: "cpu", "cuda:0"</param>
    <input>모델 경로와 장치 이름.</input>
    <output>준비된 YOLO 모델 객체. 불러오지 못하면 None.</output>
    <reason>
    첫 추론은 모델을 장치에 올리는 시간 때문에 유난히 느리다.
    카메라를 켜기 전에 미리 한 번 돌려 두면(워밍업) 첫 화면이 멈춘 듯 보이지 않는다.
    </reason>
    """
    print(f"[정보] 모델을 불러옵니다: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as error:
        print(f"[오류] 모델을 불러오지 못했습니다: {error}")
        print("인터넷 연결을 확인하거나, config.py의 MODEL_PATH가 올바른지 확인하세요.")
        return None

    print("[정보] 모델 워밍업 중...")
    dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    model.predict(dummy, device=device, imgsz=config.IMG_SIZE, verbose=False)

    return model


def color_for_class(class_id):
    """
    <summary>클래스 번호에 해당하는 색을 고른다.</summary>
    <param name="class_id">YOLO 클래스 번호(0~79).</param>
    <input>정수 클래스 번호.</input>
    <output>BGR 색 튜플.</output>
    <reason>같은 클래스는 늘 같은 색으로 그려야 프레임이 바뀌어도 한눈에 구분된다.</reason>
    """
    return PALETTE[class_id % len(PALETTE)]


def text_color_for(background):
    """
    <summary>배경색 위에서 잘 보이는 글자색(검정 또는 흰색)을 고른다.</summary>
    <param name="background">라벨 배경의 BGR 색 튜플.</param>
    <input>BGR 색.</input>
    <output>검정 또는 흰색 BGR 튜플.</output>
    <reason>밝은 배경에 흰 글자를 쓰면 읽히지 않으므로 밝기에 따라 글자색을 바꾼다.</reason>
    """
    blue, green, red = background
    brightness = 0.299 * red + 0.587 * green + 0.114 * blue

    if brightness > 150:
        return (0, 0, 0)

    return (255, 255, 255)


def draw_label(image, text, x, y, color):
    """
    <summary>상자 위쪽에 배경색이 칠해진 라벨 글자를 그린다.</summary>
    <param name="image">그림을 그릴 BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="text">라벨 문자열.</param>
    <param name="x">상자 왼쪽 위 x 좌표.</param>
    <param name="y">상자 왼쪽 위 y 좌표.</param>
    <param name="color">라벨 배경 BGR 색.</param>
    <input>이미지, 문자열, 위치, 색.</input>
    <output>없음. image가 직접 바뀐다.</output>
    <reason>상자가 화면 맨 위에 붙으면 라벨이 잘리므로 상자 안쪽으로 내려 그린다.</reason>
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_height = text_height + baseline + 4

    if y - label_height < 0:
        top = y
    else:
        top = y - label_height

    cv2.rectangle(image, (x, top), (x + text_width + 4, top + label_height), color, -1)

    text_y = top + text_height + 2
    cv2.putText(image, text, (x + 2, text_y), font, font_scale, text_color_for(color), thickness, cv2.LINE_AA)


def draw_detections(image, result):
    """
    <summary>YOLO 결과의 모든 검출 물체를 상자와 라벨로 그린다.</summary>
    <param name="image">상자를 그릴 BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="result">model.predict()가 돌려준 Results 객체 하나.</param>
    <input>이미지와 YOLO 결과.</input>
    <output>검출된 물체 수(int).</output>
    <reason>
    4단계와 달리 원본을 복사하지 않고 바로 그린다.
    실시간에서는 프레임마다 복사하는 비용도 아껴야 하고, 원본을 다시 쓸 일이 없기 때문이다.
    </reason>
    """
    names = result.names
    boxes = result.boxes

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

        color = color_for_class(class_id)
        label = f"{names[class_id]} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)

    return len(boxes)


def draw_text_lines(frame, lines):
    """
    <summary>프레임 왼쪽 위에 여러 줄의 글자를 반투명 배경과 함께 그린다.</summary>
    <param name="frame">글자를 그릴 BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="lines">그릴 문자열 목록.</param>
    <input>BGR 이미지와 문자열 목록.</input>
    <output>없음. frame이 직접 바뀐다.</output>
    <reason>cv2.putText는 한글을 못 그리므로 영어로 쓰고, 어두운 배경을 깔아 잘 읽히게 한다.</reason>
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 24
    padding = 8

    box_width = 0

    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        box_width = max(box_width, text_width)

    box_height = line_height * len(lines) + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_width + padding * 2, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

    for i, line in enumerate(lines):
        y = line_height * (i + 1)
        cv2.putText(frame, line, (padding, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def run():
    """
    <summary>웹캠 프레임마다 YOLO를 실행하고, 상자와 FPS, 추론 시간을 창에 표시한다.</summary>
    <input>config.py의 카메라, 모델, 장치, 신뢰도 설정. 키보드 입력(q).</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    모델을 카메라보다 먼저 준비한다.
    카메라를 먼저 켜면 모델 다운로드와 워밍업 동안 카메라만 켜진 채 기다리게 된다.
    </reason>
    """
    device = config.resolve_device(config.DEVICE)
    print(f"[정보] 장치: {device}")

    model = load_model(config.MODEL_PATH, device)

    if model is None:
        return 1

    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if cap is None:
        print_camera_help(config.CAMERA_INDEX)
        return 1

    fps_counter = FpsCounter()

    print("[정보] 카메라가 열렸습니다.")
    print(f"[정보] 창을 선택한 상태에서 '{config.QUIT_KEY}' 키를 누르면 종료합니다.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
                print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            results = model.predict(
                frame,
                device=device,
                conf=config.CONF_THRESHOLD,
                imgsz=config.IMG_SIZE,
                verbose=False,
            )
            result = results[0]

            count = draw_detections(frame, result)
            fps = fps_counter.update()

            lines = [
                f"FPS: {fps:.1f}",
                f"Inference: {result.speed['inference']:.0f} ms",
                f"Objects: {count}",
                f"Device: {device}",
            ]
            draw_text_lines(frame, lines)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("[정보] 종료 키를 눌러 프로그램을 끝냅니다.")
                break

            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
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
#      python step5_webcam_yolo.py
# 3. 영상 창을 클릭한 뒤 q 키를 누르면 종료한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - CPU에서 FPS는 몇 정도 나오는가? 2단계(YOLO 없음)와 비교하면?
# - 화면의 Inference 값(ms)과 FPS 사이에는 어떤 관계가 있는가?
#   (대략 FPS ≈ 1000 / 한 프레임 전체 처리 시간)
# - config.py의 IMG_SIZE를 320으로 줄이면 FPS와 인식률이 각각 어떻게 변하는가?
# - 손에 컵, 휴대폰, 책 등을 들고 비춰 보자. 어떤 물체는 잘 잡히고 어떤 물체는 안 잡히는가?
#   (YOLO 기본 모델은 COCO 데이터셋의 80개 클래스만 안다)
# - GPU가 있다면 DEVICE를 "cuda:0"으로 바꾸고 Inference 값을 비교해 보자.
