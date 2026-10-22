"""
<summary>3단계: 이미지 한 장에 YOLO 추론을 실행하고, 결과 객체 안에 무엇이 들어 있는지 살펴본다.</summary>
<reason>
웹캠처럼 계속 바뀌는 입력으로 바로 시작하면 결과를 천천히 들여다볼 수 없다.
멈춰 있는 이미지 한 장으로 "모델이 무엇을 돌려주는가"를 먼저 이해한다.
</reason>
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

import config


# 결과 이미지를 저장할 폴더
OUTPUT_DIR = Path("outputs")

# 이미지를 지정하지 않으면 ultralytics에 함께 들어 있는 예제 사진을 쓴다.
DEFAULT_IMAGE = ASSETS / "bus.jpg"


def load_model(model_path):
    """
    <summary>YOLO 모델 파일을 불러온다. 파일이 없으면 인터넷에서 내려받는다.</summary>
    <param name="model_path">모델 파일 이름 또는 경로. 예: "yolo11n.pt"</param>
    <input>모델 경로 문자열.</input>
    <output>YOLO 모델 객체. 불러오지 못하면 None.</output>
    <reason>첫 실행 때는 다운로드가 필요하므로, 인터넷 문제로 실패하면 원인을 알려 준다.</reason>
    """
    print(f"[정보] 모델을 불러옵니다: {model_path} (처음에는 다운로드에 시간이 걸릴 수 있습니다)")

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


def print_results(result):
    """
    <summary>YOLO 결과 객체에서 검출된 물체 목록과 처리 시간을 콘솔에 출력한다.</summary>
    <param name="result">model.predict()가 돌려준 목록의 원소 하나(ultralytics Results).</param>
    <input>Results 객체.</input>
    <output>없음. 콘솔에 표를 출력한다.</output>
    <reason>
    result.plot()으로 그림만 보면 내부 구조를 모른 채 넘어가게 된다.
    boxes.xyxy, boxes.conf, boxes.cls를 직접 꺼내 보아야 4단계에서 직접 그릴 수 있다.
    </reason>
    """
    names = result.names
    boxes = result.boxes

    print()
    print(f"[결과] 검출된 물체 수: {len(boxes)}")
    print(f"{'번호':>4}  {'클래스':<14}{'신뢰도':>8}   좌표 (x1, y1, x2, y2)")
    print("-" * 60)

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        print(f"{i:>4}  {names[class_id]:<14}{confidence:>8.2f}   ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    print("-" * 60)

    speed = result.speed
    print(f"[시간] 전처리 {speed['preprocess']:.1f} ms, 추론 {speed['inference']:.1f} ms, 후처리 {speed['postprocess']:.1f} ms")
    print()


def show_until_closed(window_name, image):
    """
    <summary>이미지를 창에 띄우고, 아무 키를 누르거나 창을 닫을 때까지 기다린다.</summary>
    <param name="window_name">창 제목.</param>
    <param name="image">보여 줄 BGR 이미지.</param>
    <input>창 제목과 이미지.</input>
    <output>없음.</output>
    <reason>cv2.waitKey(0)만 쓰면 X 버튼으로 창을 닫았을 때 프로그램이 멈춘 채 남을 수 있다.</reason>
    """
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(100)

        if key != -1:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break


def run():
    """
    <summary>이미지 한 장에 YOLO를 실행하고 결과를 출력, 저장, 표시한다.</summary>
    <input>명령줄 첫 번째 인자로 이미지 경로(생략하면 예제 사진). config.py의 모델, 장치, 신뢰도 설정.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>device를 predict()에 넘기면 config.py의 DEVICE 한 줄로 CPU와 GPU를 바꿀 수 있다.</reason>
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
    print(f"[정보] 이미지: {image_path}  크기: {image.shape[1]} x {image.shape[0]}")

    results = model.predict(
        image,
        device=device,
        conf=config.CONF_THRESHOLD,
        imgsz=config.IMG_SIZE,
        verbose=False,
    )

    # 이미지 한 장을 넣었으므로 결과도 한 개다.
    result = results[0]

    print_results(result)

    plotted = result.plot()

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "step3_result.jpg"
    cv2.imwrite(str(save_path), plotted)
    print(f"[정보] 결과 이미지를 저장했습니다: {save_path}")

    print("[정보] 결과 창에서 아무 키나 누르면 종료합니다.")

    try:
        show_until_closed(config.WINDOW_NAME, plotted)
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
# 2. 예제 사진(버스)으로 실행
#      python step3_yolo_image.py
# 3. 내 사진으로 실행
#      python step3_yolo_image.py my_photo.jpg
# 4. 결과는 outputs\step3_result.jpg에도 저장된다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - 콘솔 표의 좌표 (x1, y1, x2, y2)는 각각 이미지의 어디를 뜻하는가?
#   저장된 그림의 상자와 비교해 보자.
# - config.py의 CONF_THRESHOLD를 0.25, 0.8로 바꾸면 검출 수가 어떻게 변하는가?
# - [시간] 줄에서 가장 오래 걸리는 단계는 무엇인가?
#   DEVICE를 "cuda:0"으로 바꿀 수 있는 컴퓨터라면 추론 시간을 비교해 보자.
# - 처음 실행한 뒤 폴더에 yolo11n.pt 파일이 생겼는가?
#   두 번째 실행부터는 다운로드 없이 이 파일을 바로 읽는다.
