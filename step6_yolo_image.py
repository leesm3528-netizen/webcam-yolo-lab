"""
<summary>6단계: STEP 3에서 찍은 캡처 이미지에 YOLO를 적용하고, 결과 구조와 추론 속도를 살펴본다.</summary>
<reason>
움직이는 웹캠 대신 멈춘 사진 한 장으로 시작하면, 같은 입력으로 몇 번이고 결과와 속도를 비교할 수 있다.
결과 객체(boxes의 xyxy, conf, cls)를 직접 꺼내 그려 보면 STEP 7 실시간 인식에서 원하는 대로 화면을 꾸밀 수 있다.
</reason>
"""

import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils import ASSETS

import config


BASE_DIR = Path(__file__).resolve().parent

# STEP 3이 사진을 저장한 폴더
CAPTURE_DIR = BASE_DIR / "captures"

# 비교 이미지를 저장할 폴더
OUTPUT_DIR = BASE_DIR / "outputs"

# STEP 3의 캡처 파일 이름 규칙
CAPTURE_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}(_\d+)?\.jpg$")

# 캡처가 하나도 없을 때 쓸 예제 사진 (ultralytics에 함께 들어 있다)
SAMPLE_IMAGE = ASSETS / "bus.jpg"

# 이 단계의 신뢰도 기준. 이 값보다 낮은 검출은 버린다.
CONF_THRESHOLD = 0.4

# 속도 비교에 쓸 입력 크기들
BENCHMARK_SIZES = (320, 416, 640)

# 속도를 잴 때 크기마다 반복할 횟수. 여러 번 재서 평균을 내야 우연한 흔들림이 줄어든다.
BENCHMARK_REPEATS = 5

# 측정 전에 버릴 워밍업 추론 횟수. GPU는 한 번으로는 준비가 덜 끝나는 경우가 많다.
WARMUP_RUNS = 3

# 이 ultralytics 버전이 새 이름(quantize)을 쓰는지. 예전 버전은 half=True를 쓴다.
SUPPORTS_QUANTIZE = "quantize" in DEFAULT_CFG_DICT

# 비교 화면에서 패널 하나의 높이(픽셀)
PANEL_HEIGHT = 540

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
    <output>가장 최근 사진의 Path. 없으면 None.</output>
    <reason>학생이 경로를 일일이 적지 않아도 STEP 3에서 방금 찍은 사진을 바로 쓰게 한다.</reason>
    """
    if not directory.is_dir():
        return None

    captures = [p for p in directory.glob("*.jpg") if CAPTURE_NAME_PATTERN.match(p.name)]

    if not captures:
        return None

    return max(captures, key=capture_sort_key)


def load_input_image():
    """
    <summary>실습에 쓸 이미지 1장을 준비한다. 명령줄 경로 → 최근 캡처 → 예제 사진 순서로 시도한다.</summary>
    <input>명령줄 첫 번째 인자(선택)로 이미지 경로.</input>
    <output>(BGR 이미지, 출처 설명 문자열) 튜플. 읽지 못하면 (None, None).</output>
    <reason>
    캡처가 없어도 예제 사진으로 실습을 이어 갈 수 있게 한다.
    cv2.imread는 실패해도 예외 없이 None만 돌려주므로 반드시 확인한다.
    </reason>
    """
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        source = str(path)
    else:
        path = find_latest_capture(CAPTURE_DIR)

        if path is not None:
            source = f"{path.name} (STEP 3 최근 캡처)"
        else:
            print("[안내] captures 폴더에 사진이 없어 예제 사진(bus.jpg)을 씁니다.")
            print("       STEP 3(step3_capture.py)에서 사람, 컵, 휴대폰 등이 보이게 찍어 두면 더 좋습니다.")
            path = SAMPLE_IMAGE
            source = "bus.jpg (ultralytics 예제)"

    image = cv2.imread(str(path))

    if image is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {path}")
        print("경로가 맞는지, 경로에 한글이 없는지, jpg/png 파일인지 확인하세요.")
        return None, None

    return image, source


def load_model(model_path):
    """
    <summary>YOLO 모델 파일을 불러온다. 파일이 없으면 인터넷에서 내려받는다.</summary>
    <param name="model_path">모델 파일 이름 또는 경로. 예: "yolo26n.pt"</param>
    <input>모델 경로 문자열.</input>
    <output>YOLO 모델 객체. 불러오지 못하면 None.</output>
    <reason>
    최신 모델(yolo26n)은 ultralytics 버전이 낮으면 열리지 않거나, 인터넷 문제로 내려받지 못할 수 있다.
    그럴 때 무엇을 바꾸면 되는지(yolo11n.pt) 바로 알려 준다.
    </reason>
    """
    print(f"[정보] 모델을 불러옵니다: {model_path} (처음에는 다운로드에 시간이 걸릴 수 있습니다)")

    try:
        model = YOLO(model_path)
    except Exception as error:
        print(f"[오류] 모델을 불러오지 못했습니다: {error}")
        print("다음을 확인하세요.")
        print("  1. 인터넷에 연결되어 있는가? (첫 실행 때 모델 파일을 내려받는다)")
        print("  2. ultralytics가 최신인가?  pip install -U ultralytics")
        print(f"  3. 그래도 안 되면 config.py의 MODEL_PATH를 \"{config.FALLBACK_MODEL_PATH}\"로 바꿔 다시 실행한다.")
        return None

    return model


def precision_options(half):
    """
    <summary>FP16(half) 사용 여부를 predict()에 넘길 인자로 바꾼다.</summary>
    <param name="half">True면 16비트 실수(FP16)로 계산한다.</param>
    <input>FP16 여부.</input>
    <output>predict()에 덧붙일 인자 딕셔너리. FP32면 빈 딕셔너리.</output>
    <reason>
    ultralytics 8.4부터 half=True는 quantize=16이라는 새 이름으로 바뀌었고, 옛 이름을 쓰면 매번 경고가 뜬다.
    설치된 버전에 맞는 이름을 골라 넘기면, 예전 버전을 쓰는 학생 컴퓨터에서도 같은 코드가 동작한다.
    FP32일 때는 아무 인자도 넘기지 않는다. half=False만 넘겨도 경고가 뜨기 때문이다.
    </reason>
    """
    if not half:
        return {}

    if SUPPORTS_QUANTIZE:
        return {"quantize": 16}

    return {"half": True}


def predict(model, image, imgsz, device, half):
    """
    <summary>이미지 1장에 YOLO를 실행하고, 전체 걸린 시간(벽시계 시간)도 함께 잰다.</summary>
    <param name="model">YOLO 모델 객체.</param>
    <param name="image">BGR 이미지(numpy 배열).</param>
    <param name="imgsz">모델 입력 크기. 이미지를 이 크기에 맞춰 줄여서 넣는다.</param>
    <param name="device">추론 장치 이름. 예: "cpu", "cuda:0"</param>
    <param name="half">True면 16비트 실수(FP16)로 계산한다. GPU에서만 쓴다.</param>
    <input>모델, 이미지, 입력 크기, 장치, FP16 여부.</input>
    <output>(Results 객체, 전체 시간 ms) 튜플.</output>
    <reason>
    results[0].speed는 전처리·추론·후처리 각 구간만 잰 값이다.
    predict() 호출 전체를 perf_counter로 따로 재면, 세 구간 바깥의 준비 시간까지 포함한 실제 체감 시간을 알 수 있다.
    OpenCV로 읽은 numpy 배열은 BGR로 보고 ultralytics가 안에서 RGB로 바꿔 주므로 그대로 넘긴다(STEP 4 참고).
    </reason>
    """
    start = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False,
        **precision_options(half),
    )

    total_ms = (time.perf_counter() - start) * 1000

    return results[0], total_ms


def warm_up(model, image, imgsz, device, half):
    """
    <summary>측정 전에 추론을 한 번 실행해 둔다. 이 결과는 버린다.</summary>
    <param name="model">YOLO 모델 객체.</param>
    <param name="image">BGR 이미지.</param>
    <param name="imgsz">모델 입력 크기.</param>
    <param name="device">추론 장치 이름.</param>
    <param name="half">FP16 여부.</param>
    <input>predict()와 같은 인자.</input>
    <output>첫 번째 워밍업 추론에 걸린 시간(ms).</output>
    <reason>
    첫 추론은 모델을 장치(GPU 메모리)에 올리고, 계산 방법을 고르고, 메모리를 잡는 일까지 함께 한다.
    GPU는 입력 크기마다 가장 빠른 계산 방법을 몇 번 시험해 보므로, 한 번으로는 준비가 덜 끝나기도 한다.
    이 시간을 평소 속도로 착각하지 않도록 처음 WARMUP_RUNS번은 측정에서 뺀다.
    </reason>
    """
    _, first_ms = predict(model, image, imgsz, device, half)

    for _ in range(WARMUP_RUNS - 1):
        predict(model, image, imgsz, device, half)

    return first_ms


def verify_half_precision(model, image, device, sizes):
    """
    <summary>FP16 결과가 FP32 결과와 같은지(탐지 수가 같은지) 크기별로 확인한다.</summary>
    <param name="model">YOLO 모델 객체.</param>
    <param name="image">BGR 이미지.</param>
    <param name="device">GPU 장치 이름.</param>
    <param name="sizes">확인할 입력 크기 목록.</param>
    <input>모델, 이미지, 장치, 크기 목록.</input>
    <output>모든 크기에서 같으면 True, 하나라도 다르면 False.</output>
    <reason>
    FP16은 숫자를 표현하는 자릿수가 적어, 일부 GPU(특히 GTX 16 시리즈)에서는 특정 입력 크기에서 결과가 통째로 사라지기도 한다.
    오류 없이 "0개 탐지"로만 나타나서 알아채기 어렵기 때문에, 측정 전에 FP32 결과와 맞춰 본다.
    </reason>
    """
    all_same = True

    for imgsz in sizes:
        warm_up(model, image, imgsz, device, False)
        fp32, _ = predict(model, image, imgsz, device, False)

        warm_up(model, image, imgsz, device, True)
        fp16, _ = predict(model, image, imgsz, device, True)

        same = len(fp32.boxes) == len(fp16.boxes)
        mark = "같음" if same else "다름!"
        print(f"         imgsz={imgsz:<4} FP32 {len(fp32.boxes)}개 / FP16 {len(fp16.boxes)}개  → {mark}")

        if not same:
            all_same = False

    return all_same


def print_detection_table(result, names):
    """
    <summary>results[0].boxes에서 xyxy, conf, cls를 꺼내 표로 출력한다.</summary>
    <param name="result">model.predict()가 돌려준 Results 객체 하나.</param>
    <param name="names">클래스 번호 → 이름 딕셔너리(model.names).</param>
    <input>Results 객체와 클래스 이름표.</input>
    <output>없음. 콘솔에 표를 출력한다.</output>
    <reason>
    boxes.xyxy는 (x1, y1, x2, y2) = (왼쪽 위 x, 왼쪽 위 y, 오른쪽 아래 x, 오른쪽 아래 y)다.
    값은 GPU에 있을 수도 있는 텐서이므로 .cpu().numpy()로 꺼내야 numpy처럼 다룰 수 있다.
    cls는 실수(0.0, 2.0...)로 들어 있어서 int()로 바꿔야 names에서 이름을 찾을 수 있다.
    </reason>
    """
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    print()
    print("=" * 78)
    print(f"검출 결과  (conf ≥ {CONF_THRESHOLD}, {len(boxes)}개)")
    print("=" * 78)

    if len(boxes) == 0:
        print("검출된 물체가 없습니다. 물체가 크게 보이게 다시 찍거나 CONF_THRESHOLD를 낮춰 보세요.")
        return

    print(f"{'#':>2}  {'cls':>3}  {'name':<14}{'conf':>6}   {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}   {'w':>5} {'h':>5}")
    print("-" * 78)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        class_id = classes[i]

        print(
            f"{i:>2}  {class_id:>3}  {names[class_id]:<14}{confs[i]:>6.2f}   "
            f"{x1:>5.0f} {y1:>5.0f} {x2:>5.0f} {y2:>5.0f}   {x2 - x1:>5.0f} {y2 - y1:>5.0f}"
        )


def print_speed_breakdown(result, total_ms):
    """
    <summary>results[0].speed의 전처리·추론·후처리 시간과 각 구간의 비중을 출력한다.</summary>
    <param name="result">Results 객체.</param>
    <param name="total_ms">predict() 호출 전체를 잰 시간(ms).</param>
    <input>Results 객체와 전체 시간.</input>
    <output>없음. 콘솔에 표를 출력한다.</output>
    <reason>
    "YOLO가 느리다"고 할 때 실제로 어느 구간이 느린지 나눠 봐야 무엇을 줄일지 정할 수 있다.
    전처리는 크기 조절과 색 변환, 추론은 신경망 계산, 후처리는 겹친 상자를 정리하는 단계다.
    </reason>
    """
    speed = result.speed
    parts = [
        ("전처리 preprocess", speed["preprocess"]),
        ("추론   inference", speed["inference"]),
        ("후처리 postprocess", speed["postprocess"]),
    ]
    measured = sum(ms for _, ms in parts)

    print()
    print("=" * 78)
    print("구간별 시간 (results[0].speed)")
    print("=" * 78)

    for label, ms in parts:
        share = ms / measured * 100 if measured > 0 else 0
        bar = "#" * round(share / 2.5)
        print(f"{label:<20}{ms:>8.1f} ms  {share:>5.1f}%  {bar}")

    print("-" * 78)
    print(f"{'세 구간 합계':<20}{measured:>8.1f} ms")
    print(f"{'predict() 전체':<20}{total_ms:>8.1f} ms  (구간 밖 준비 시간 {max(total_ms - measured, 0):.1f} ms 포함)")


def color_for_class(class_id):
    """
    <summary>클래스 번호에 해당하는 색을 고른다.</summary>
    <param name="class_id">YOLO 클래스 번호(0~79).</param>
    <input>정수 클래스 번호.</input>
    <output>BGR 색 튜플.</output>
    <reason>같은 클래스는 늘 같은 색으로 그려야 한눈에 구분된다.</reason>
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


def draw_detections(image, result, names):
    """
    <summary>cv2.rectangle과 cv2.putText로 검출 상자와 "이름 신뢰도" 라벨을 직접 그린다.</summary>
    <param name="image">원본 BGR 이미지. 이 배열은 바뀌지 않는다.</param>
    <param name="result">Results 객체.</param>
    <param name="names">클래스 번호 → 이름 딕셔너리.</param>
    <input>원본 이미지, 결과, 클래스 이름표.</input>
    <output>상자가 그려진 새 BGR 이미지.</output>
    <reason>
    STEP 5의 draw_info_panel과 같은 방법이다. getTextSize로 글자 크기를 재고, 두께 -1 상자로 라벨 배경을 칠한다.
    좌표가 실수이므로 OpenCV에 넘기기 전에 int로 바꾼다.
    상자가 이미지 맨 위에 붙으면 라벨이 잘리므로 그때는 상자 안쪽에 그린다.
    </reason>
    """
    canvas = image.copy()
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_width = max(2, round(min(canvas.shape[:2]) / 300))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        class_id = classes[i]
        color = color_for_class(class_id)
        label = f"{names[class_id]} {confs[i]:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_height = text_height + baseline + 6

        if y1 - label_height < 0:
            top = y1
        else:
            top = y1 - label_height

        cv2.rectangle(canvas, (x1, top), (x1 + text_width + 8, top + label_height), color, -1)
        cv2.putText(canvas, label, (x1 + 4, top + text_height + 3), font, font_scale, text_color_for(color), thickness, cv2.LINE_AA)

    return canvas


def add_title(image, title):
    """
    <summary>패널 위쪽에 제목 띠를 붙인다.</summary>
    <param name="image">BGR 이미지.</param>
    <param name="title">띠에 쓸 영어 제목.</param>
    <input>이미지와 제목.</input>
    <output>제목 띠가 위에 붙은 새 BGR 이미지.</output>
    <reason>두 결과를 나란히 놓았을 때 어느 쪽이 무엇인지 바로 알 수 있게 한다.</reason>
    """
    bar = np.full((36, image.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

    return cv2.vconcat([bar, image])


def resize_to_height(image, height):
    """
    <summary>가로세로 비율을 유지하면서 높이를 맞춘다.</summary>
    <param name="image">BGR 이미지.</param>
    <param name="height">맞출 높이(픽셀).</param>
    <input>이미지와 목표 높이.</input>
    <output>크기가 바뀐 이미지.</output>
    <reason>큰 사진 두 장을 붙이면 화면을 넘치므로 줄인다. cv2.resize는 (가로, 세로) 순서로 받는다.</reason>
    """
    old_height, old_width = image.shape[:2]
    new_width = round(old_width * height / old_height)

    return cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)


def build_comparison(image, result, names):
    """
    <summary>직접 그린 결과와 results[0].plot() 결과를 가로로 붙인다.</summary>
    <param name="image">원본 BGR 이미지.</param>
    <param name="result">Results 객체.</param>
    <param name="names">클래스 번호 → 이름 딕셔너리.</param>
    <input>원본 이미지, 결과, 클래스 이름표.</input>
    <output>두 패널을 가로로 붙인 BGR 이미지.</output>
    <reason>같은 결과를 두 방법으로 그려 겹쳐 보면, 직접 꺼낸 좌표가 맞는지 눈으로 검증할 수 있다.</reason>
    """
    mine = draw_detections(image, result, names)
    official = result.plot()

    left = add_title(resize_to_height(mine, PANEL_HEIGHT), "My drawing (cv2.rectangle / putText)")
    right = add_title(resize_to_height(official, PANEL_HEIGHT), "Ultralytics (results[0].plot())")

    return cv2.hconcat([left, right])


def compare_image_sizes(model, image, device, half, sizes=BENCHMARK_SIZES, repeats=BENCHMARK_REPEATS):
    """
    <summary>입력 크기(imgsz)를 바꿔 가며 속도와 탐지 수를 비교한다.</summary>
    <param name="model">YOLO 모델 객체.</param>
    <param name="image">BGR 이미지.</param>
    <param name="device">추론 장치 이름.</param>
    <param name="half">FP16 여부.</param>
    <param name="sizes">비교할 입력 크기 목록.</param>
    <param name="repeats">크기마다 반복해서 잴 횟수.</param>
    <input>모델, 이미지, 장치, FP16 여부, 크기 목록, 반복 횟수.</input>
    <output>크기별 결과 딕셔너리 목록. 키: imgsz, total_ms, preprocess, inference, postprocess, count, classes.</output>
    <reason>
    imgsz가 작으면 신경망이 계산할 칸이 줄어 빨라지지만, 작은 물체는 몇 픽셀로 뭉개져 놓치기 쉽다.
    크기마다 처음 WARMUP_RUNS번은 워밍업으로 버리고 나머지를 평균 낸다.
    크기가 바뀌면 내부 준비를 다시 하는 경우가 있어 첫 추론이 유난히 느리기 때문이다.
    </reason>
    """
    rows = []

    for imgsz in sizes:
        warm_up(model, image, imgsz, device, half)

        totals = []
        pre = []
        inf = []
        post = []
        result = None

        for _ in range(repeats):
            result, total_ms = predict(model, image, imgsz, device, half)
            totals.append(total_ms)
            pre.append(result.speed["preprocess"])
            inf.append(result.speed["inference"])
            post.append(result.speed["postprocess"])

        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = sorted({model.names[c] for c in classes})

        rows.append({
            "imgsz": imgsz,
            "total_ms": sum(totals) / repeats,
            "preprocess": sum(pre) / repeats,
            "inference": sum(inf) / repeats,
            "postprocess": sum(post) / repeats,
            "count": len(result.boxes),
            "classes": names,
        })

    return rows


def print_size_comparison(rows, device, half):
    """
    <summary>compare_image_sizes()의 결과를 표로 출력한다.</summary>
    <param name="rows">compare_image_sizes()가 돌려준 목록.</param>
    <param name="device">추론 장치 이름.</param>
    <param name="half">FP16 여부.</param>
    <input>결과 목록, 장치, FP16 여부.</input>
    <output>없음. 콘솔에 표를 출력한다.</output>
    <reason>가장 큰 크기(640)를 기준으로 몇 배 빠른지 함께 보여 주면 속도와 정확도의 맞바꿈이 한눈에 보인다.</reason>
    """
    base_ms = rows[-1]["total_ms"]

    print()
    print("=" * 78)
    print(f"입력 크기별 비교  (device={device}, half={half}, 크기마다 워밍업 {WARMUP_RUNS}회 + 측정 {BENCHMARK_REPEATS}회 평균)")
    print("=" * 78)
    print(f"{'imgsz':>5}  {'전체':>8}  {'전처리':>7}  {'추론':>7}  {'후처리':>7}  {'FPS':>5}  {'640대비':>6}  {'탐지':>4}  클래스")
    print("-" * 78)

    for row in rows:
        fps = 1000 / row["total_ms"] if row["total_ms"] > 0 else 0
        speedup = base_ms / row["total_ms"] if row["total_ms"] > 0 else 0

        print(
            f"{row['imgsz']:>5}  {row['total_ms']:>6.1f}ms  {row['preprocess']:>5.1f}ms  {row['inference']:>5.1f}ms  "
            f"{row['postprocess']:>5.1f}ms  {fps:>5.1f}  {speedup:>5.2f}x  {row['count']:>4}  {', '.join(row['classes'])}"
        )


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
            return

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return


def run():
    """
    <summary>이미지 준비 → 모델 준비 → 워밍업 → 측정 추론 → 표·구간 시간 출력 → 그림 비교 → 크기별 비교 순으로 실행한다.</summary>
    <input>명령줄 이미지 경로(선택). config.py의 MODEL_PATH, IMG_SIZE, DEVICE.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    GPU(cuda)에서는 half=True로 16비트 실수를 써서 메모리를 절반으로 줄이고 속도를 높인다.
    CPU는 16비트 계산이 오히려 느리거나 지원되지 않으므로 32비트 그대로 둔다.
    단, FP16 결과가 FP32와 다르면 빠른 것보다 맞는 것이 먼저이므로 FP32로 되돌린다.
    </reason>
    """
    image, source = load_input_image()

    if image is None:
        return 1

    model = load_model(config.MODEL_PATH)

    if model is None:
        return 1

    device = config.resolve_device(config.DEVICE)
    half = device.startswith("cuda")

    print()
    print(f"[정보] 이미지 : {source}  ({image.shape[1]} x {image.shape[0]})")
    print(f"[정보] 모델   : {config.MODEL_PATH}  (클래스 {len(model.names)}개)")
    print(f"[정보] 장치   : {device}  half={half}")

    if half:
        print("[정보] FP16(half) 결과가 FP32와 같은지 확인합니다...")
        check_sizes = sorted(set(BENCHMARK_SIZES) | {config.IMG_SIZE})

        if not verify_half_precision(model, image, device, check_sizes):
            print("[안내] 이 GPU에서는 FP16 결과가 FP32와 달라 half=False(FP32)로 진행합니다.")
            print("       일부 GPU(특히 GTX 16 시리즈)는 FP16 계산이 불안정해 탐지가 사라질 수 있습니다.")
            half = False

    warmup_ms = warm_up(model, image, config.IMG_SIZE, device, half)
    print(f"[정보] 워밍업 : 첫 추론 {warmup_ms:.1f} ms  (처음 {WARMUP_RUNS}회는 측정에서 제외)")

    result, total_ms = predict(model, image, config.IMG_SIZE, device, half)
    print(f"[정보] 추론   : {total_ms:.1f} ms  (imgsz={config.IMG_SIZE}, 워밍업 이후)")

    print_detection_table(result, model.names)
    print_speed_breakdown(result, total_ms)

    comparison = build_comparison(image, result, model.names)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "step6_compare.jpg"
    cv2.imwrite(str(save_path), comparison)

    print()
    print("[정보] 입력 크기별 속도를 재는 중입니다...")
    rows = compare_image_sizes(model, image, device, half)
    print_size_comparison(rows, device, half)

    print()
    print(f"[정보] 비교 이미지를 저장했습니다: {save_path}")
    print("[정보] 결과 창에서 아무 키나 누르면 종료합니다.")

    try:
        show_until_closed("Step6 - YOLO on capture", comparison)
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
# 2. STEP 3에서 찍은 최근 사진으로 실행 (없으면 예제 사진 bus.jpg)
#      python step6_yolo_image.py
# 3. 특정 사진으로 실행
#      python step6_yolo_image.py captures\20261022_141530.jpg
# 4. GPU로 실행하려면 config.py에서 DEVICE = "cuda:0"으로 바꾼다.
#    half=True(FP16)가 자동 적용된다. ultralytics 8.4부터는 같은 뜻의 quantize=16으로 넘긴다.
#    FP16 결과가 FP32와 다르게 나오는 GPU에서는 안내 문구와 함께 FP32로 되돌린다.
# 5. 결과는 outputs\step6_compare.jpg에도 저장된다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 전처리·추론·후처리 시간의 비중 (results[0].speed)
#    - "구간별 시간" 표에서 가장 큰 비중을 차지하는 구간은 어디인가?
#      전처리: 이미지를 imgsz에 맞게 줄이고, BGR→RGB, 0~255 → 0~1로 바꾸는 일
#      추론  : 신경망 계산 (대부분의 연산이 여기에 있다)
#      후처리: 겹친 상자 정리, 좌표를 원본 크기로 되돌리기
#    - CPU에서는 보통 추론 비중이 가장 크다. GPU로 바꾸면 비중이 어떻게 달라지는가?
#      (추론이 크게 빨라지면, 상대적으로 전처리와 후처리 비중이 커진다.
#       "GPU를 써도 생각만큼 안 빨라진다"면 이 구간들을 의심해야 한다)
#    - 원본 사진이 클수록(예: 4000 x 3000 휴대폰 사진) 어느 구간이 늘어나는가?
#
# 2. 워밍업
#    - [정보] 워밍업의 첫 추론 시간과 그다음 추론 시간을 비교해 보자. 몇 배 차이 나는가?
#      첫 추론을 빼지 않고 재면 어떤 착각을 하게 될까?
#    - WARMUP_RUNS를 1로 줄이고 GPU에서 실행해 보자. 크기별 비교 표에서 작은 imgsz가 오히려 느리게 나오는가?
#      (GPU는 크기마다 계산 방법을 시험하느라 처음 몇 번이 느리다)
#
# 3. FP16(half)과 FP32
#    - GPU에서 "FP16 결과가 FP32와 같은지" 확인 결과는 어떤가?
#      FP16은 빠르고 메모리를 덜 쓰지만, 자릿수가 적어 GPU에 따라 결과가 사라지기도 한다.
#      "빠른데 틀린 결과"보다 "조금 느려도 맞는 결과"가 먼저다.
#
# 4. 입력 크기(imgsz) 320 / 416 / 640
#    - 640 대비 320은 몇 배 빠른가? 계산할 칸 수는 (640/320)² = 4배 차이다. 실제 측정값과 비교해 보자.
#    - GPU에서는 320과 640의 차이가 CPU보다 훨씬 작다(거의 같을 수도 있다). 왜 그럴까?
#      (GPU는 수천 개의 계산을 동시에 처리하므로 n 모델의 640도 일감이 모자라다.
#       그러면 계산량보다 "GPU에 일을 맡기고 결과를 받아오는 고정 시간"이 시간을 좌우한다)
#    - 탐지 수는 어떻게 변하는가? 멀리 있는 작은 물체부터 사라지는가?
#    - STEP 7 실시간 인식에서 CPU로 FPS를 높이고 싶다면 어떤 imgsz를 고를지 표를 보고 정해 보자.
#
# 5. 결과 구조
#    - 왼쪽(직접 그림)과 오른쪽(plot)의 상자 위치는 같은가?
#    - 표의 w, h(가로, 세로 길이)는 x2 - x1, y2 - y1이다. 가장 큰 물체는 무엇인가?
#    - 이 사진에서 물체만 잘라내려면 image[y1:y2, x1:x2]로 쓴다. 왜 y가 먼저일까? (STEP 4 좌표계)
