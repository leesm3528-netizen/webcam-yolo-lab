"""
<summary>4단계: STEP 3에서 저장한 캡처 이미지 1장으로 OpenCV 이미지의 기본(배열 정보, 색 순서, 좌표계)을 익힌다.</summary>
<reason>
YOLO에 넣는 것은 결국 numpy 배열이다.
배열의 모양(shape), 값의 종류(dtype), 색 순서(BGR/RGB), 좌표 순서((x, y)와 [y, x])를 모르면
모델 입력을 잘못 넣어도 오류 없이 결과만 틀려져 원인을 찾기 어렵다.
</reason>
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config


BASE_DIR = Path(__file__).resolve().parent

# STEP 3이 사진을 저장한 폴더
CAPTURE_DIR = BASE_DIR / "captures"

# 결과 비교 이미지를 저장할 폴더
OUTPUT_DIR = BASE_DIR / "outputs"

# STEP 3의 캡처 파일 이름 규칙. 예: 20261022_141530.jpg, 20261022_141530_2.jpg
CAPTURE_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}(_\d+)?\.jpg$")

# 비교 화면에서 각 패널의 높이(픽셀). 세 장을 가로로 붙여도 화면에 들어오게 줄인다.
PANEL_HEIGHT = 360

# 웹캠으로 찍을 때 버릴 첫 프레임 수. 카메라가 밝기와 초점을 맞출 시간을 준다.
WARMUP_FRAMES = 15

# 좌표계 설명에 쓸 예시 점 (x, y)
SAMPLE_X = 200
SAMPLE_Y = 80


def capture_sort_key(path):
    """
    <summary>캡처 파일을 촬영 순서대로 정렬하기 위한 기준값을 만든다.</summary>
    <param name="path">규칙에 맞는 캡처 파일 경로.</param>
    <input>Path 객체.</input>
    <output>(시각 문자열, 같은 초 안의 번호) 튜플.</output>
    <reason>글자 순서로만 정렬하면 "_10"이 "_2"보다 앞에 오므로, 번호는 숫자로 비교한다.</reason>
    """
    parts = path.stem.split("_")
    stamp = parts[0] + parts[1]

    if len(parts) == 3:
        number = int(parts[2])
    else:
        number = 1

    return stamp, number


def find_latest_capture(directory):
    """
    <summary>captures 폴더에서 가장 최근에 찍은 사진을 찾는다.</summary>
    <param name="directory">캡처 사진이 들어 있는 폴더.</param>
    <input>폴더 경로.</input>
    <output>가장 최근 사진의 Path. 폴더가 없거나 사진이 없으면 None.</output>
    <reason>STEP 3에서 찍은 사진을 학생이 경로를 일일이 적지 않아도 바로 쓰게 한다.</reason>
    """
    if not directory.is_dir():
        return None

    captures = [p for p in directory.glob("*.jpg") if CAPTURE_NAME_PATTERN.match(p.name)]

    if not captures:
        return None

    return max(captures, key=capture_sort_key)


def capture_from_webcam(index, width, height):
    """
    <summary>웹캠에서 사진 1장을 찍는다.</summary>
    <param name="index">카메라 번호.</param>
    <param name="width">요청할 가로 해상도.</param>
    <param name="height">요청할 세로 해상도.</param>
    <input>카메라 번호와 해상도.</input>
    <output>BGR 이미지(numpy 배열). 카메라를 열지 못하거나 프레임을 못 읽으면 None.</output>
    <reason>
    켜자마자 찍은 첫 프레임은 어둡거나 초록빛이 도는 경우가 많다.
    처음 몇 장을 버리고 찍어야 색 비교 실습에 쓸 만한 사진이 된다.
    어떤 경우든 finally에서 카메라를 해제한다.
    </reason>
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    try:
        if not cap.isOpened():
            print(f"[오류] {index}번 카메라를 열 수 없습니다.")
            print("다른 프로그램이 카메라를 쓰고 있지 않은지, Windows 카메라 권한이 켜져 있는지 확인하세요.")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("[정보] 카메라를 준비하는 중입니다. 카메라를 바라봐 주세요...")

        frame = None

        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()

            if not ok:
                frame = None

            time.sleep(0.03)

        if frame is None:
            print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
            print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
            return None

        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        return frame

    finally:
        cap.release()


def save_capture(frame, directory):
    """
    <summary>웹캠으로 찍은 사진을 STEP 3과 같은 이름 규칙으로 captures 폴더에 저장한다.</summary>
    <param name="frame">저장할 BGR 이미지.</param>
    <param name="directory">저장할 폴더.</param>
    <input>BGR 이미지와 폴더 경로.</input>
    <output>저장한 파일의 Path. 실패하면 None.</output>
    <reason>대신 찍은 사진도 저장해 두면 다음 단계에서 같은 사진으로 실습을 이어 갈 수 있다.</reason>
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        print(f"[오류] 사진을 저장하지 못했습니다: {path}")
        return None

    return path


def load_input_image():
    """
    <summary>실습에 쓸 이미지 1장을 준비한다. 명령줄 경로 → 최근 캡처 → 웹캠 촬영 순서로 시도한다.</summary>
    <input>명령줄 첫 번째 인자(선택)로 이미지 경로.</input>
    <output>(BGR 이미지, 출처 설명 문자열) 튜플. 모두 실패하면 (None, None).</output>
    <reason>
    STEP 3을 건너뛴 학생도 이 단계를 바로 실습할 수 있도록 웹캠 촬영으로 대신한다.
    cv2.imread는 파일이 없거나 깨져도 예외 없이 None을 돌려주므로 반드시 확인한다.
    </reason>
    """
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        image = cv2.imread(str(path))

        if image is None:
            print(f"[오류] 이미지를 읽을 수 없습니다: {path}")
            print("경로가 맞는지, 경로에 한글이 없는지, jpg/png 파일인지 확인하세요.")
            return None, None

        return image, str(path)

    path = find_latest_capture(CAPTURE_DIR)

    if path is not None:
        image = cv2.imread(str(path))

        if image is not None:
            return image, f"{path.name} (STEP 3 최근 캡처)"

        print(f"[안내] 최근 캡처를 읽지 못했습니다: {path}")

    print("[안내] captures 폴더에 사진이 없어 웹캠으로 1장을 찍습니다.")
    image = capture_from_webcam(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if image is None:
        return None, None

    saved = save_capture(image, CAPTURE_DIR)

    if saved is None:
        return image, "웹캠 촬영 (저장 실패)"

    print(f"[정보] 찍은 사진을 저장했습니다: {saved}")
    return image, f"{saved.name} (웹캠 촬영)"


def print_image_info(image, source):
    """
    <summary>이미지 배열의 모양, 자료형, 픽셀값 범위를 출력한다.</summary>
    <param name="image">BGR 이미지(numpy 배열).</param>
    <param name="source">이미지 출처 설명.</param>
    <input>BGR 이미지와 출처 문자열.</input>
    <output>없음. 콘솔에 정보를 출력한다.</output>
    <reason>
    shape가 (세로, 가로, 채널) 순서라는 것과 dtype이 uint8(0~255 정수)이라는 것은
    이후 모든 단계(크기 조절, 색 변환, YOLO 입력)의 전제가 된다.
    </reason>
    """
    height, width, channels = image.shape

    print()
    print("=" * 60)
    print("1. 이미지 정보")
    print("=" * 60)
    print(f"출처            : {source}")
    print(f"type            : {type(image).__name__}  (OpenCV 이미지는 numpy 배열이다)")
    print(f"shape           : {image.shape}  → (세로 {height}, 가로 {width}, 채널 {channels})")
    print(f"dtype           : {image.dtype}  → 0~255 정수 한 칸(1바이트)")
    print(f"전체 값 개수    : {image.size:,}  = {height} x {width} x {channels}")
    print(f"메모리          : {image.nbytes / 1024:.0f} KB")
    print(f"픽셀값 범위     : 최소 {image.min()}, 최대 {image.max()}")

    for i, name in enumerate(["B(파랑)", "G(초록)", "R(빨강)"]):
        channel = image[:, :, i]
        print(f"  채널 {i} {name} : 최소 {channel.min():>3}, 최대 {channel.max():>3}, 평균 {channel.mean():6.1f}")


def convert_colors(bgr):
    """
    <summary>BGR 이미지를 RGB와 그레이스케일로 바꾼다.</summary>
    <param name="bgr">BGR 이미지.</param>
    <input>BGR 이미지(세로 x 가로 x 3).</input>
    <output>(RGB 이미지, 그레이스케일 이미지) 튜플. 그레이스케일은 세로 x 가로(2차원)다.</output>
    <reason>
    cvtColor는 채널 순서만 바꿀 뿐 픽셀값 자체는 바꾸지 않는다(RGB의 경우).
    그레이스케일은 세 채널을 가중 평균해 한 채널로 줄이므로 배열 차원이 하나 줄어든다.
    </reason>
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    return rgb, gray


def print_pixel_comparison(bgr, rgb, gray, x, y):
    """
    <summary>같은 위치의 픽셀값을 BGR, RGB, 그레이스케일에서 각각 꺼내 비교한다.</summary>
    <param name="bgr">BGR 이미지.</param>
    <param name="rgb">RGB 이미지.</param>
    <param name="gray">그레이스케일 이미지.</param>
    <param name="x">가로 위치.</param>
    <param name="y">세로 위치.</param>
    <input>세 이미지와 좌표.</input>
    <output>없음. 콘솔에 비교 결과를 출력한다.</output>
    <reason>
    숫자로 직접 보면 RGB 변환이 "첫 번째와 세 번째 값을 맞바꾼 것"뿐임을 확인할 수 있다.
    그레이 값이 0.299R + 0.587G + 0.114B 공식과 맞는지도 계산해 본다.
    </reason>
    """
    b, g, r = [int(v) for v in bgr[y, x]]
    expected_gray = 0.299 * r + 0.587 * g + 0.114 * b

    print()
    print("=" * 60)
    print(f"2. 색 순서 비교 (x={x}, y={y} 위치의 픽셀)")
    print("=" * 60)
    print(f"bgr[{y}, {x}]  = {bgr[y, x].tolist()}  → [B, G, R]")
    print(f"rgb[{y}, {x}]  = {rgb[y, x].tolist()}  → [R, G, B]  (첫 값과 끝 값이 바뀌었다)")
    print(f"gray[{y}, {x}] = {int(gray[y, x])}  → 값 1개 (shape {gray.shape}, 채널 없음)")
    print(f"공식 계산     = 0.299*{r} + 0.587*{g} + 0.114*{b} = {expected_gray:.1f}")


def resize_to_height(image, height):
    """
    <summary>가로세로 비율을 유지하면서 이미지 높이를 맞춘다.</summary>
    <param name="image">BGR 또는 그레이스케일 이미지.</param>
    <param name="height">맞출 높이(픽셀).</param>
    <input>이미지와 목표 높이.</input>
    <output>크기가 바뀐 이미지.</output>
    <reason>
    cv2.hconcat은 모든 이미지의 높이가 같아야 붙일 수 있다.
    cv2.resize는 크기를 (가로, 세로) 순서로 받는다. shape의 (세로, 가로)와 순서가 반대라는 점에 주의한다.
    </reason>
    """
    old_height, old_width = image.shape[:2]
    new_width = round(old_width * height / old_height)

    return cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)


def to_three_channels(image):
    """
    <summary>그레이스케일(1채널) 이미지를 3채널로 바꾼다. 이미 3채널이면 그대로 돌려준다.</summary>
    <param name="image">이미지(2차원 또는 3차원 배열).</param>
    <input>이미지.</input>
    <output>세로 x 가로 x 3 이미지.</output>
    <reason>
    cv2.hconcat은 채널 수와 dtype도 같아야 붙일 수 있다.
    GRAY2BGR은 같은 값을 세 채널에 복사할 뿐이라 겉보기는 여전히 흑백이다.
    </reason>
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def add_title(image, title, subtitle):
    """
    <summary>이미지 위쪽에 두 줄짜리 제목 띠를 붙인다.</summary>
    <param name="image">3채널 이미지.</param>
    <param name="title">첫 줄 제목(영어).</param>
    <param name="subtitle">둘째 줄 설명(영어).</param>
    <input>이미지와 문자열 두 개.</input>
    <output>제목 띠가 위에 붙은 새 이미지.</output>
    <reason>cv2.putText는 한글을 못 그리므로 영어로 쓴다. 띠를 붙인 뒤에도 패널끼리 높이가 같게 유지된다.</reason>
    """
    bar = np.full((52, image.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    return cv2.vconcat([bar, image])


def build_color_comparison(bgr, rgb, gray):
    """
    <summary>BGR, RGB, 그레이스케일 이미지를 크기와 채널을 맞춰 가로로 붙인다.</summary>
    <param name="bgr">BGR 이미지.</param>
    <param name="rgb">RGB 이미지.</param>
    <param name="gray">그레이스케일 이미지.</param>
    <input>세 이미지.</input>
    <output>세 패널을 가로로 붙인 BGR 이미지 한 장.</output>
    <reason>
    cv2.imshow는 배열을 늘 BGR이라고 생각하고 화면에 그린다.
    그래서 RGB 배열을 그대로 넘기면 빨강과 파랑이 뒤바뀐 색으로 보인다. 이것이 바로 색 순서 실수의 모습이다.
    </reason>
    """
    panels = [
        (bgr, "1) BGR (OpenCV default)", "imshow(bgr): correct colors"),
        (rgb, "2) RGB array shown as-is", "imshow(rgb): red and blue swapped"),
        (gray, "3) Grayscale (1 ch -> 3 ch)", f"shape {gray.shape} -> GRAY2BGR"),
    ]

    prepared = []

    for image, title, subtitle in panels:
        resized = resize_to_height(image, PANEL_HEIGHT)
        three = to_three_channels(resized)
        prepared.append(add_title(three, title, subtitle))

    for i, panel in enumerate(prepared):
        print(f"  패널 {i + 1}: shape {panel.shape}, dtype {panel.dtype}")

    return cv2.hconcat(prepared)


def demonstrate_coordinates(bgr, x, y):
    """
    <summary>OpenCV 그리기 함수의 (x, y) 순서와 numpy 배열 인덱스의 [y, x] 순서 차이를 보여 준다.</summary>
    <param name="bgr">BGR 이미지.</param>
    <param name="x">예시 점의 가로 위치.</param>
    <param name="y">예시 점의 세로 위치.</param>
    <input>BGR 이미지와 예시 좌표.</input>
    <output>설명이 그려진 새 BGR 이미지.</output>
    <reason>
    cv2.circle, cv2.rectangle, cv2.resize 같은 OpenCV 함수는 (x, y) = (가로, 세로) 순서를 쓴다.
    numpy 배열은 행(세로)부터 고르므로 image[y, x] 순서다.
    YOLO 상자 좌표 (x1, y1, x2, y2)로 이미지를 잘라낼 때 image[y1:y2, x1:x2]로 써야 하는 것도 같은 이유다.
    </reason>
    """
    canvas = bgr.copy()
    height, width = canvas.shape[:2]

    print()
    print("=" * 60)
    print("3. 좌표계: (x, y)와 [y, x]")
    print("=" * 60)
    print(f"이미지 크기     : 가로 {width}, 세로 {height}")
    print(f"shape           : {canvas.shape}  → shape[0]이 세로, shape[1]이 가로")
    print("원점 (0, 0)     : 왼쪽 위. x는 오른쪽으로, y는 아래로 커진다.")
    print()

    # (1) OpenCV 그리기 함수: (x, y) 순서
    cv2.circle(canvas, (x, y), 8, (0, 0, 255), 2)
    print(f"cv2.circle(img, (x={x}, y={y}), ...)  → 가로 {x}, 세로 {y} 위치에 원")

    # (2) numpy 인덱스: [y, x] 순서
    print(f"img[y={y}, x={x}]                    = {bgr[y, x].tolist()}  ← 원 중심의 픽셀")

    if x < height and y < width:
        print(f"img[{x}, {y}] (순서를 바꾸면)          = {bgr[x, y].tolist()}  ← 전혀 다른 위치(가로 {y}, 세로 {x})의 픽셀")
        cv2.circle(canvas, (y, x), 8, (255, 0, 255), 2)
    else:
        print(f"img[{x}, {y}] (순서를 바꾸면)          → IndexError! 세로가 {height}까지밖에 없다")

    # (3) 슬라이싱: [y 범위, x 범위]
    box_x, box_y = x + 40, y
    box_w, box_h = 120, 40
    canvas[box_y:box_y + box_h, box_x:box_x + box_w] = (0, 200, 255)
    print()
    print(f"img[{box_y}:{box_y + box_h}, {box_x}:{box_x + box_w}] = 주황색")
    print(f"  → 세로 {box_h}픽셀, 가로 {box_w}픽셀짜리 납작한 직사각형이 칠해진다.")
    print(f"     같은 영역을 cv2.rectangle로 그리면 (x1, y1)=({box_x}, {box_y}), (x2, y2)=({box_x + box_w}, {box_y + box_h})")

    # (4) 크기 지정: cv2.resize는 (가로, 세로)
    small = cv2.resize(bgr, (320, 180))
    print()
    print(f"cv2.resize(img, (320, 180)).shape = {small.shape}  → 넘길 때는 (가로, 세로), shape는 (세로, 가로)")

    # 화면 설명 글자
    labels = [
        (f"(x={x}, y={y})  cv2.circle(img, (x, y))", (0, 0, 255)),
        (f"img[{x}, {y}]  swapped order -> different pixel", (255, 0, 255)),
        (f"img[{box_y}:{box_y + box_h}, {box_x}:{box_x + box_w}]  rows=y, cols=x", (0, 200, 255)),
    ]

    # 글자 뒤에 어두운 띠를 깔아 밝은 배경에서도 읽히게 한다.
    # (굵은 검정 글자 위에 얇은 색 글자를 겹치는 방식은 OpenCV 5에서 글자 폭이 달라 어긋난다)
    line_height = 24
    band_top = height - 16 - len(labels) * line_height + 4
    band = canvas.copy()
    cv2.rectangle(band, (0, band_top), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.6, canvas, 0.4, 0, dst=canvas)

    for i, (text, color) in enumerate(labels):
        text_y = height - 16 - (len(labels) - 1 - i) * line_height
        cv2.putText(canvas, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 축 방향 표시
    cv2.arrowedLine(canvas, (10, 10), (90, 10), (255, 255, 255), 2, tipLength=0.2)
    cv2.arrowedLine(canvas, (10, 10), (10, 90), (255, 255, 255), 2, tipLength=0.2)
    cv2.putText(canvas, "x", (96, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y", (6, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def show_windows_until_closed(windows):
    """
    <summary>여러 창을 띄우고, 아무 키를 누르거나 창 하나를 닫을 때까지 기다린다.</summary>
    <param name="windows">(창 제목, 이미지) 튜플 목록.</param>
    <input>창 목록.</input>
    <output>없음.</output>
    <reason>cv2.waitKey(0)만 쓰면 X 버튼으로 창을 닫았을 때 프로그램이 멈춘 채 남을 수 있다.</reason>
    """
    for name, image in windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)

    while True:
        key = cv2.waitKey(100)

        if key != -1:
            return

        for name, _ in windows:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return


def run():
    """
    <summary>이미지 1장을 준비해 정보 출력, 색 변환 비교, 좌표계 설명을 차례로 보여 준다.</summary>
    <input>명령줄 이미지 경로(선택). 없으면 captures 폴더의 최근 사진, 그것도 없으면 웹캠 촬영.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 이미지를 준비하지 못하면 1.</output>
    <reason>콘솔의 숫자와 창의 그림을 함께 보면, 배열 값이 화면에 어떻게 나타나는지 연결해서 이해할 수 있다.</reason>
    """
    bgr, source = load_input_image()

    if bgr is None:
        print("[오류] 실습에 쓸 이미지를 준비하지 못했습니다.")
        print("STEP 3(step3_capture.py)에서 사진을 먼저 찍거나, 이미지 경로를 직접 넘겨 주세요.")
        return 1

    height, width = bgr.shape[:2]
    x = min(SAMPLE_X, width - 1)
    y = min(SAMPLE_Y, height - 1)

    print_image_info(bgr, source)

    rgb, gray = convert_colors(bgr)
    print_pixel_comparison(bgr, rgb, gray, x, y)

    print()
    print("[정보] hconcat 전에 각 패널의 크기와 채널을 맞춥니다.")
    color_view = build_color_comparison(bgr, rgb, gray)
    coord_view = demonstrate_coordinates(bgr, x, y)

    OUTPUT_DIR.mkdir(exist_ok=True)
    color_path = OUTPUT_DIR / "step4_basics_colors.jpg"
    coord_path = OUTPUT_DIR / "step4_basics_coords.jpg"
    cv2.imwrite(str(color_path), color_view)
    cv2.imwrite(str(coord_path), coord_view)

    print()
    print(f"[정보] 비교 이미지를 저장했습니다: {color_path}")
    print(f"[정보] 좌표 설명 이미지를 저장했습니다: {coord_path}")
    print("[정보] 창에서 아무 키나 누르면 종료합니다.")

    try:
        show_windows_until_closed([
            ("Step4 - BGR / RGB / Gray", color_view),
            ("Step4 - Coordinates", coord_view),
        ])
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. STEP 3에서 찍은 최근 사진으로 실행
#      python step4_opencv_basics.py
# 3. 특정 사진으로 실행
#      python step4_opencv_basics.py captures\20261022_141530.jpg
# 4. captures 폴더가 비어 있으면 웹캠으로 1장을 찍어 captures에 저장한 뒤 그 사진을 쓴다.
# 5. 콘솔 출력을 먼저 읽고, 두 창을 비교한다. 아무 키나 누르면 종료한다.
#    결과는 outputs\step4_basics_colors.jpg, outputs\step4_basics_coords.jpg에도 저장된다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. OpenCV가 BGR을 쓰는 이유
#    - OpenCV가 처음 만들어진 2000년 무렵, 카메라 장치와 Windows 비트맵(BMP)이 픽셀을 B, G, R 바이트 순서로
#      저장하는 경우가 많았다. 그 순서를 그대로 따른 것이 지금까지 기본값으로 남았다.
#    - 반면 matplotlib, PIL(Pillow), 웹 브라우저, 대부분의 딥러닝 학습 데이터는 RGB 순서다.
#    - 2번 패널에서 얼굴이나 피부색이 파랗게 보이는가? 콘솔의 bgr[y, x]와 rgb[y, x] 값을 비교해
#      어느 값이 자리를 바꿨는지 찾아보자.
#
# 2. 왜 YOLO 입력 전에 색 순서가 중요한가
#    - YOLO는 RGB 사진으로 학습되었다. 색이 뒤바뀐 입력을 넣으면 오류 없이 "조금 틀린 결과"만 나온다.
#      주황색 공이 파란 공처럼 보이고, 신뢰도가 떨어지거나 다른 클래스로 잘못 인식될 수 있다.
#    - ultralytics는 numpy 배열을 받으면 "OpenCV에서 온 BGR"이라고 보고 안에서 RGB로 바꿔 준다.
#      그래서 cv2.imread나 cap.read()로 얻은 배열은 그대로 넣으면 된다.
#      이미 RGB로 바꾼 배열을 넣으면 한 번 더 뒤집혀서 오히려 틀린 색이 된다.
#    - 직접 확인해 보기: 아래처럼 RGB 배열을 일부러 넣고 STEP 3의 결과(신뢰도)와 비교해 보자.
#        from ultralytics import YOLO
#        model = YOLO("yolo11n.pt")
#        print(model.predict(bgr, verbose=False)[0].boxes.conf)   # 올바른 입력
#        print(model.predict(rgb, verbose=False)[0].boxes.conf)   # 색이 뒤바뀐 입력
#
# 3. 좌표계
#    - 빨간 원(cv2.circle에 (x, y))과 보라 원(img[x, y]로 순서를 바꿔 읽은 위치)은 같은 곳에 있는가?
#    - 주황 직사각형은 왜 세로로 긴 모양이 아니라 가로로 긴 모양인가?
#    - YOLO 상자 (x1, y1, x2, y2)로 물체만 잘라내려면 img[?:?, ?:?]에 무엇을 넣어야 할까?
#
# 4. hconcat의 조건
#    - build_color_comparison에서 to_three_channels를 빼면 어떤 오류가 나는가? 직접 지워 보고 되돌려 놓자.
#    - 콘솔에 출력된 패널 세 개의 shape는 모두 같은가?
