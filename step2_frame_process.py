"""
<summary>2단계: 웹캠 프레임을 가공(좌우 반전, 크기 조절, 흑백)하고 FPS를 화면에 표시한다.</summary>
<reason>
YOLO에 들어가는 것은 결국 "한 장의 이미지(numpy 배열)"다.
프레임을 직접 바꿔 보면서 프레임이 배열이라는 감각과, 처리 속도(FPS)를 재는 방법을 먼저 익힌다.
</reason>
"""

import sys
import time

import cv2

import config


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
    <param name="smoothing">이전 FPS를 얼마나 유지할지 정하는 값(0~1). 클수록 숫자가 천천히 변한다.</param>
    <input>매 프레임마다 update()를 호출한다.</input>
    <output>update()가 부드럽게 다듬은 FPS 값을 돌려준다.</output>
    <reason>
    프레임마다 1/간격을 그대로 보여 주면 숫자가 심하게 흔들려 읽기 어렵다.
    지수 이동 평균으로 다듬으면 흐름을 한눈에 볼 수 있다.
    </reason>
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
        <input>없음. 호출한 시각을 기준으로 계산한다.</input>
        <output>다듬어진 FPS 값(float).</output>
        <reason>첫 값은 그대로 쓰고 이후에만 평균을 내야 처음에 0에서 천천히 올라가는 현상이 없다.</reason>
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


def process_frame(frame, flip, gray, scale):
    """
    <summary>프레임에 좌우 반전, 크기 조절, 흑백 변환을 차례로 적용한다.</summary>
    <param name="frame">카메라에서 읽은 BGR 이미지(numpy 배열, 모양은 세로 x 가로 x 3).</param>
    <param name="flip">True면 좌우를 뒤집는다.</param>
    <param name="gray">True면 흑백으로 바꾼다.</param>
    <param name="scale">크기 배율. 1.0이면 원래 크기다.</param>
    <input>BGR 이미지와 처리 옵션.</input>
    <output>처리된 BGR 이미지(numpy 배열).</output>
    <reason>
    흑백으로 바꾼 뒤 다시 3채널로 되돌리는 이유는, 그 위에 컬러 글자를 그리기 위해서다.
    1채널 이미지에는 초록색 같은 색을 칠할 수 없다.
    </reason>
    """
    if flip:
        frame = cv2.flip(frame, 1)

    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if gray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    return frame


def draw_text_lines(frame, lines):
    """
    <summary>프레임 왼쪽 위에 여러 줄의 글자를 반투명 배경과 함께 그린다.</summary>
    <param name="frame">글자를 그릴 BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="lines">그릴 문자열 목록. 한 원소가 한 줄이다.</param>
    <input>BGR 이미지와 문자열 목록.</input>
    <output>없음. frame이 직접 바뀐다.</output>
    <reason>
    cv2.putText는 한글을 그리지 못하므로 화면 글자는 영어로 쓴다.
    배경 없이 글자만 그리면 밝은 장면에서 읽히지 않아 어두운 상자를 깐다.
    </reason>
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
    <summary>웹캠 영상을 가공해 표시하고, 키보드로 가공 옵션을 바꾼다.</summary>
    <input>config.py의 카메라 설정. 키보드 입력(f, g, +, -, q).</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 카메라 오류는 1.</output>
    <reason>옵션을 실행 중에 켜고 끄면 처리 전후 차이와 FPS 변화를 바로 비교할 수 있다.</reason>
    """
    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if cap is None:
        print_camera_help(config.CAMERA_INDEX)
        return 1

    flip = config.FLIP_HORIZONTAL
    gray = False
    scale = 1.0
    fps_counter = FpsCounter()

    print("[정보] 카메라가 열렸습니다.")
    print("[조작] f: 좌우 반전  g: 흑백  +/-: 크기 조절  q: 종료")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
                print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                break

            frame = process_frame(frame, flip, gray, scale)
            fps = fps_counter.update()

            height, width = frame.shape[:2]

            lines = [
                f"FPS: {fps:.1f}",
                f"Size: {width} x {height} (x{scale:.2f})",
                f"[f] Flip: {'ON' if flip else 'OFF'}",
                f"[g] Gray: {'ON' if gray else 'OFF'}",
            ]
            draw_text_lines(frame, lines)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("[정보] 종료 키를 눌러 프로그램을 끝냅니다.")
                break

            if key == ord("f"):
                flip = not flip

            if key == ord("g"):
                gray = not gray

            # + 키는 Shift를 눌러야 하므로 같은 자리의 = 키도 받아 준다.
            if key in (ord("+"), ord("=")):
                scale = min(scale + 0.25, 2.0)

            if key == ord("-"):
                scale = max(scale - 0.25, 0.25)

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
#      python step2_frame_process.py
# 3. 영상 창을 클릭한 뒤 키를 누른다.
#      f: 좌우 반전 켜기/끄기
#      g: 흑백 켜기/끄기
#      + 또는 =: 크기 키우기
#      -: 크기 줄이기
#      q: 종료
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - f를 눌러 반전을 끄면 손을 오른쪽으로 움직였을 때 화면에서는 어느 쪽으로 움직이는가?
# - 크기를 x2.00으로 키우면 FPS가 어떻게 변하는가? x0.25로 줄이면?
#   (이미지가 클수록 처리할 픽셀이 많아진다. YOLO에서도 똑같이 적용된다)
# - 흑백으로 바꾸었는데도 FPS 글자는 왜 초록색으로 보이는가?
#   (process_frame의 docstring에 있는 reason을 읽어 보자)
# - 카메라 자체의 FPS(보통 30)보다 화면 FPS가 더 높게 나올 수 있는가?
