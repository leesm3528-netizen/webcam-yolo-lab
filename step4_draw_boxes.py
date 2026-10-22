"""
<summary>4단계: YOLO 결과에서 좌표, 클래스, 신뢰도를 꺼내 OpenCV로 상자와 라벨을 직접 그린다.</summary>
<reason>
result.plot()은 편리하지만 색, 글자, 표시할 물체를 마음대로 바꿀 수 없다.
직접 그려 보면 결과 구조를 확실히 이해하고, 이후 단계에서 원하는 정보를 덧붙일 수 있다.
</reason>
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

import config


OUTPUT_DIR = Path("outputs")

DEFAULT_IMAGE = ASSETS / "bus.jpg"

# 클래스 번호마다 돌아가며 쓸 색 목록 (BGR 순서)
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


def load_model(model_path):
    """
    <summary>YOLO 모델 파일을 불러온다. 파일이 없으면 인터넷에서 내려받는다.</summary>
    <param name="model_path">모델 파일 이름 또는 경로. 예: "yolo11n.pt"</param>
    <input>모델 경로 문자열.</input>
    <output>YOLO 모델 객체. 불러오지 못하면 None.</output>
    <reason>첫 실행 때는 다운로드가 필요하므로, 인터넷 문제로 실패하면 원인을 알려 준다.</reason>
    """
    print(f"[정보] 모델을 불러옵니다: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as error:
        print(f"[오류] 모델을 불러오지 못했습니다: {error}")
        print("인터넷 연결을 확인하거나, config.py의 MODEL_PATH가 올바른지 확인하세요.")
        return None

    return model


def load_image(path):
    """
    <summary>이미지 파일을 BGR 배열로 읽는다.</summary>
    <param name="path">이미지 파일 경로.</param>
    <input>파일 경로.</input>
    <output>BGR 이미지(numpy 배열). 읽지 못하면 None.</output>
    <reason>cv2.imread는 실패해도 예외 없이 None만 돌려주므로 직접 확인해야 한다.</reason>
    """
    image = cv2.imread(str(path))

    if image is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {path}")
        print("경로가 맞는지, 한글이 없는 경로인지, jpg/png 파일인지 확인하세요.")
        return None

    return image


def color_for_class(class_id):
    """
    <summary>클래스 번호에 해당하는 색을 고른다.</summary>
    <param name="class_id">YOLO 클래스 번호(0~79).</param>
    <input>정수 클래스 번호.</input>
    <output>BGR 색 튜플.</output>
    <reason>같은 클래스는 늘 같은 색으로 그려야 화면이 바뀌어도 한눈에 구분된다.</reason>
    """
    return PALETTE[class_id % len(PALETTE)]


def text_color_for(background):
    """
    <summary>배경색 위에서 잘 보이는 글자색(검정 또는 흰색)을 고른다.</summary>
    <param name="background">라벨 배경의 BGR 색 튜플.</param>
    <input>BGR 색.</input>
    <output>검정 (0, 0, 0) 또는 흰색 (255, 255, 255).</output>
    <reason>밝은 노란 배경에 흰 글자를 쓰면 읽히지 않으므로 밝기에 따라 글자색을 바꾼다.</reason>
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
    <param name="text">라벨 문자열. 예: "person 0.87"</param>
    <param name="x">상자 왼쪽 위 x 좌표.</param>
    <param name="y">상자 왼쪽 위 y 좌표.</param>
    <param name="color">라벨 배경 BGR 색.</param>
    <input>이미지, 문자열, 위치, 색.</input>
    <output>없음. image가 직접 바뀐다.</output>
    <reason>
    상자가 이미지 맨 위에 붙어 있으면 라벨이 화면 밖으로 잘린다.
    그럴 때는 라벨을 상자 안쪽으로 내려 그린다.
    </reason>
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
    <param name="image">원본 BGR 이미지. 이 배열은 바뀌지 않는다.</param>
    <param name="result">model.predict()가 돌려준 Results 객체 하나.</param>
    <input>원본 이미지와 YOLO 결과.</input>
    <output>상자가 그려진 새 BGR 이미지.</output>
    <reason>
    원본을 복사해서 그려야, 같은 원본으로 result.plot()과 나란히 비교할 수 있다.
    boxes.xyxy는 실수(float)이므로 OpenCV에 넘기기 전에 정수로 바꾼다.
    </reason>
    """
    canvas = image.copy()
    names = result.names
    boxes = result.boxes

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

        color = color_for_class(class_id)
        label = f"{names[class_id]} {confidence:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        draw_label(canvas, label, x1, y1, color)

    return canvas


def add_title(image, title):
    """
    <summary>이미지 위쪽에 제목 띠를 붙인 새 이미지를 만든다.</summary>
    <param name="image">BGR 이미지.</param>
    <param name="title">띠에 쓸 영어 제목.</param>
    <input>이미지와 제목 문자열.</input>
    <output>제목 띠가 위에 붙은 새 BGR 이미지.</output>
    <reason>두 그림을 나란히 놓았을 때 어느 쪽이 무엇인지 바로 알 수 있게 한다.</reason>
    """
    bar = np.zeros((32, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([bar, image])


def show_until_closed(window_name, image):
    """
    <summary>이미지를 창에 띄우고, 아무 키를 누르거나 창을 닫을 때까지 기다린다.</summary>
    <param name="window_name">창 제목.</param>
    <param name="image">보여 줄 BGR 이미지.</param>
    <input>창 제목과 이미지.</input>
    <output>없음.</output>
    <reason>cv2.waitKey(0)만 쓰면 X 버튼으로 창을 닫았을 때 프로그램이 멈춘 채 남을 수 있다.</reason>
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(100)

        if key != -1:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break


def run():
    """
    <summary>이미지에 YOLO를 실행하고, 직접 그린 결과와 result.plot() 결과를 나란히 보여 준다.</summary>
    <input>명령줄 첫 번째 인자로 이미지 경로(생략하면 예제 사진). config.py의 모델, 장치, 신뢰도 설정.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>두 결과를 np.hstack으로 붙여 보면 직접 그린 코드가 맞게 동작하는지 바로 검증된다.</reason>
    """
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = DEFAULT_IMAGE

    image = load_image(image_path)

    if image is None:
        return 1

    model = load_model(config.MODEL_PATH)

    if model is None:
        return 1

    device = config.resolve_device(config.DEVICE)
    print(f"[정보] 장치: {device}")

    results = model.predict(
        image,
        device=device,
        conf=config.CONF_THRESHOLD,
        imgsz=config.IMG_SIZE,
        verbose=False,
    )
    result = results[0]

    print(f"[결과] 검출된 물체 수: {len(result.boxes)}")

    mine = add_title(draw_detections(image, result), "My drawing (draw_detections)")
    official = add_title(result.plot(), "Ultralytics (result.plot)")
    side_by_side = np.hstack([mine, official])

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "step4_compare.jpg"
    cv2.imwrite(str(save_path), side_by_side)
    print(f"[정보] 비교 이미지를 저장했습니다: {save_path}")

    print("[정보] 결과 창에서 아무 키나 누르면 종료합니다.")

    try:
        show_until_closed(config.WINDOW_NAME, side_by_side)
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
# 2. 예제 사진으로 실행
#      python step4_draw_boxes.py
# 3. 내 사진으로 실행
#      python step4_draw_boxes.py my_photo.jpg
# 4. 결과는 outputs\step4_compare.jpg에도 저장된다.
#    창이 화면보다 크면 창 모서리를 끌어서 크기를 줄일 수 있다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - 왼쪽(직접 그림)과 오른쪽(result.plot)의 상자 위치가 정확히 같은가?
# - 같은 클래스(예: person)는 모두 같은 색으로 그려졌는가?
# - draw_detections에서 int()를 빼면 어떤 오류가 나는가? 직접 지워 보고 되돌려 놓자.
# - 이미지 맨 위에 걸친 물체의 라벨은 어디에 그려지는가? (draw_label의 reason)
# - 특정 클래스만 그리고 싶다면 draw_detections의 어디를 고치면 될까?
#   예: names[class_id] == "person"일 때만 그리기
