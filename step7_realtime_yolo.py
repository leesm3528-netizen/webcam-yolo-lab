"""
<summary>7단계(최종): 웹캠 실시간 YOLO 객체 인식. cv2 창 모드와 Flask 웹 모드를 지원하고, 장치에 맞춰 속도 설정을 자동으로 고른다.</summary>
<reason>
STEP 1~6에서 익힌 카메라 읽기, OpenCV 그리기, Flask 스트리밍, YOLO 추론을 하나로 합친다.
CPU 노트북에서도 실시간에 가깝게 돌도록 입력 크기(imgsz)를 줄이고 몇 프레임마다 한 번만 추론(skip)한다.
명령줄 옵션으로 설정을 바꿔 가며 FPS를 재 보면, 속도와 정확도의 맞바꿈을 숫자로 확인할 수 있다.
</reason>
"""

import argparse
import atexit
import re
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils import ASSETS

import config


BASE_DIR = Path(__file__).resolve().parent

# 탐지 결과 프레임을 저장할 폴더.
# STEP 3의 captures 폴더 안 하위 폴더로 두어, 원본 캡처(STEP 4·6의 실습 재료)와 섞이지 않게 한다.
DETECTION_DIR = BASE_DIR / "captures" / "detected"

# 웹 모드 서버 주소
HOST = "127.0.0.1"
PORT = 5000

# 카메라가 연속으로 이 횟수만큼 프레임을 못 읽으면 끊긴 것으로 본다.
MAX_READ_FAILURES = 30

# 웹 페이지에 보여 줄 최근 촬영 수
HISTORY_LIMIT = 10

# 저장 파일 이름 규칙. 예: 20261022_141530.jpg, 같은 초에 두 번 찍으면 20261022_141530_2.jpg
CAPTURE_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}(_\d+)?\.jpg$")

# 측정 전에 버릴 워밍업 추론 횟수
WARMUP_RUNS = 3

# FPS를 평균 낼 최근 프레임 수
FPS_WINDOW = 30

# 기본 신뢰도 기준
DEFAULT_CONF = 0.4

# 장치 종류별 권장값. 명령줄에서 직접 지정하면 그 값이 이긴다.
CPU_DEFAULTS = {"imgsz": 320, "skip": 2}
GPU_DEFAULTS = {"imgsz": 640, "skip": 1}

# 이 ultralytics 버전이 FP16 옵션의 새 이름(quantize)을 쓰는지. 예전 버전은 half=True를 쓴다.
SUPPORTS_QUANTIZE = "quantize" in DEFAULT_CFG_DICT

WINDOW_NAME = "Step7 - Realtime YOLO"

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


# ============================================================
# 1. 명령줄 옵션과 실행 설정
# ============================================================

def parse_args(argv=None):
    """
    <summary>명령줄 옵션을 읽는다.</summary>
    <param name="argv">옵션 문자열 목록. None이면 실제 명령줄(sys.argv)을 쓴다.</param>
    <input>명령줄 문자열.</input>
    <output>argparse.Namespace 객체.</output>
    <reason>
    imgsz와 skip의 기본값을 None으로 두는 이유는 "학생이 직접 정했는지"를 구분하기 위해서다.
    None이면 장치에 맞는 권장값을 자동으로 채우고, 값을 적었으면 그대로 존중한다.
    </reason>
    """
    parser = argparse.ArgumentParser(
        description="STEP 7: 웹캠 실시간 YOLO 객체 인식",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["window", "web"], default="window", help="window: cv2 창, web: 브라우저")
    parser.add_argument("--device", choices=["cpu", "cuda:0", "mps"], default=config.DEVICE, help="추론 장치")
    parser.add_argument("--imgsz", type=int, default=None, help="모델 입력 크기 (생략하면 CPU 320, GPU 640)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="신뢰도 기준 (0~1)")
    parser.add_argument("--skip", type=int, default=None, help="N프레임마다 한 번 추론 (생략하면 CPU 2, GPU 1)")
    parser.add_argument("--model", default=config.MODEL_PATH, help="모델 파일 (.pt)")
    parser.add_argument("--openvino", action="store_true", help="<모델이름>_openvino_model 폴더가 있으면 OpenVINO로 실행 (Intel CPU 가속)")
    parser.add_argument("--classes", nargs="+", default=None, metavar="NAME", help="이 클래스만 탐지. 예: --classes person cup")

    args = parser.parse_args(argv)

    if args.imgsz is not None and args.imgsz < 32:
        parser.error("--imgsz는 32 이상이어야 합니다. (보통 320, 416, 640)")

    if args.skip is not None and args.skip < 1:
        parser.error("--skip은 1 이상이어야 합니다. (1이면 모든 프레임을 추론)")

    if not 0.0 < args.conf < 1.0:
        parser.error("--conf는 0과 1 사이여야 합니다. 예: 0.4")

    return args


@dataclass
class RunSettings:
    """
    <summary>실제로 쓸 실행 설정 묶음. 명령줄 값과 자동 권장값을 합친 결과다.</summary>
    <param name="device">추론 장치 이름.</param>
    <param name="imgsz">모델 입력 크기.</param>
    <param name="skip">몇 프레임마다 한 번 추론할지.</param>
    <param name="conf">신뢰도 기준.</param>
    <param name="half">FP16 사용 여부.</param>
    <param name="model_path">불러올 모델 경로(.pt 파일 또는 OpenVINO 폴더).</param>
    <param name="openvino">OpenVINO 모델을 쓰는지.</param>
    <param name="notes">설정마다 "직접 지정"인지 "자동"인지 적어 둔 설명.</param>
    <input>resolve_settings()가 만든다.</input>
    <output>Detector, FrameProcessor가 읽어 쓴다.</output>
    <reason>설정을 한 객체로 묶으면 함수마다 인자를 여러 개 넘기지 않아도 되고, 화면과 웹에 같은 값을 보여 주기 쉽다.</reason>
    """

    device: str
    imgsz: int
    skip: int
    conf: float
    half: bool
    model_path: str
    openvino: bool
    notes: dict = field(default_factory=dict)

    def device_label(self):
        """
        <summary>화면에 보여 줄 장치 설명 문자열을 만든다.</summary>
        <input>없음.</input>
        <output>예: "cpu", "cpu OpenVINO", "cuda:0 FP16"</output>
        <reason>같은 cpu라도 OpenVINO 여부, 같은 cuda라도 FP16 여부에 따라 속도가 크게 달라서 함께 표시한다.</reason>
        """
        label = self.device

        if self.openvino:
            label += " OpenVINO"

        if self.half:
            label += " FP16"

        return label


def find_openvino_model(model_path):
    """
    <summary>모델 파일 옆에 OpenVINO로 변환한 폴더(예: yolo26n_openvino_model)가 있는지 찾는다.</summary>
    <param name="model_path">원래 모델 경로. 예: "yolo26n.pt"</param>
    <input>모델 경로 문자열.</input>
    <output>폴더가 있으면 그 경로 문자열, 없으면 None.</output>
    <reason>ultralytics는 export(format="openvino")를 하면 "모델이름_openvino_model" 폴더를 만든다. 그 이름 규칙을 그대로 따른다.</reason>
    """
    path = Path(model_path)
    folder = path.parent / f"{path.stem}_openvino_model"

    if folder.is_dir():
        return str(folder)

    return None


def resolve_settings(args):
    """
    <summary>명령줄 옵션과 장치별 권장값을 합쳐 실제 실행 설정을 정한다.</summary>
    <param name="args">parse_args()가 돌려준 Namespace.</param>
    <input>명령줄 옵션.</input>
    <output>RunSettings 객체.</output>
    <reason>
    CPU는 계산이 느리므로 imgsz를 320으로 줄이고 2프레임에 한 번만 추론해야 실시간에 가깝다.
    GPU(cuda)는 여유가 있으므로 640, 매 프레임 추론, FP16(half)로 정확도와 반응성을 함께 챙긴다.
    OpenVINO는 Intel CPU용 가속이므로 cpu 장치일 때만 쓴다.
    </reason>
    """
    device = config.resolve_device(args.device)
    is_gpu = device != "cpu"

    if is_gpu:
        defaults = GPU_DEFAULTS
        default_note = f"{device} 권장값"
    else:
        defaults = CPU_DEFAULTS
        default_note = "CPU 권장값"

    notes = {}

    if args.device != device:
        notes["device"] = f"요청 {args.device} → 사용할 수 없어 cpu"
    else:
        notes["device"] = "명령줄 또는 config.py"

    if args.imgsz is None:
        imgsz = defaults["imgsz"]
        notes["imgsz"] = f"자동 ({default_note})"
    else:
        imgsz = args.imgsz
        notes["imgsz"] = "직접 지정"

    if args.skip is None:
        skip = defaults["skip"]
        notes["skip"] = f"자동 ({default_note})"
    else:
        skip = args.skip
        notes["skip"] = "직접 지정"

    # FP16은 NVIDIA GPU(cuda)에서만 켠다. CPU는 16비트 계산이 오히려 느리거나 지원되지 않는다.
    half = device.startswith("cuda")

    if half:
        notes["half"] = "자동 (cuda)"
    else:
        notes["half"] = "FP32"

    model_path = args.model
    openvino = False

    if args.openvino:
        if is_gpu:
            print("[안내] --openvino는 Intel CPU 가속용이라 GPU 장치에서는 쓰지 않습니다. .pt 모델로 실행합니다.")
        else:
            folder = find_openvino_model(args.model)

            if folder is None:
                stem = Path(args.model).stem
                print(f"[안내] {stem}_openvino_model 폴더가 없어 .pt 모델로 실행합니다.")
                print("       OpenVINO 모델은 아래 두 명령으로 한 번만 만들어 두면 됩니다.")
                print("         pip install openvino")
                print(f"         yolo export model={args.model} format=openvino")
            else:
                model_path = folder
                openvino = True

    if openvino:
        notes["model"] = "OpenVINO 폴더"
    else:
        notes["model"] = "PyTorch (.pt)"

    return RunSettings(
        device=device,
        imgsz=imgsz,
        skip=skip,
        conf=args.conf,
        half=half,
        model_path=model_path,
        openvino=openvino,
        notes=notes,
    )


def resolve_class_filter(class_names, model_names):
    """
    <summary>--classes로 받은 이름을 YOLO 클래스 번호 목록으로 바꾼다.</summary>
    <param name="class_names">명령줄에서 받은 클래스 이름 목록. 없으면 None.</param>
    <param name="model_names">클래스 번호 → 이름 딕셔너리(model.names).</param>
    <input>이름 목록과 모델의 클래스 이름표.</input>
    <output>(클래스 번호 목록 또는 None, 찾지 못한 이름 목록) 튜플. None은 "전체 클래스"를 뜻한다.</output>
    <reason>
    predict()의 classes 인자는 이름이 아니라 번호를 받는다.
    대소문자는 무시하고, "cell phone"처럼 띄어쓰기가 있는 이름은 cell_phone으로 적어도 되게 한다.
    </reason>
    """
    if not class_names:
        return None, []

    lookup = {name.lower(): class_id for class_id, name in model_names.items()}
    ids = []
    unknown = []

    for raw in class_names:
        key = raw.lower().replace("_", " ")

        if key in lookup:
            ids.append(lookup[key])
        else:
            unknown.append(raw)

    return sorted(set(ids)), unknown


def print_settings(settings, class_ids, model_names):
    """
    <summary>실제로 적용된 설정을 표로 출력한다.</summary>
    <param name="settings">RunSettings 객체.</param>
    <param name="class_ids">탐지할 클래스 번호 목록. None이면 전체.</param>
    <param name="model_names">클래스 번호 → 이름 딕셔너리.</param>
    <input>설정, 클래스 필터, 이름표.</input>
    <output>없음. 콘솔에 표를 출력한다.</output>
    <reason>자동으로 바뀐 값이 있으면 학생이 모르고 지나치기 쉬우므로, 값마다 출처를 함께 보여 준다.</reason>
    """
    if class_ids is None:
        classes_text = f"전체 ({len(model_names)}개)"
    else:
        classes_text = ", ".join(model_names[i] for i in class_ids)

    rows = [
        ("장치", settings.device, settings.notes.get("device", "")),
        ("모델", settings.model_path, settings.notes.get("model", "")),
        ("imgsz", settings.imgsz, settings.notes.get("imgsz", "")),
        ("skip", settings.skip, settings.notes.get("skip", "")),
        ("half", settings.half, settings.notes.get("half", "")),
        ("conf", settings.conf, ""),
        ("classes", classes_text, ""),
    ]

    print()
    print("=" * 64)
    print("실행 설정")
    print("=" * 64)

    for name, value, note in rows:
        print(f"{name:<8} {str(value):<28} {note}")

    print("=" * 64)


# ============================================================
# 2. YOLO 탐지
# ============================================================

@dataclass
class Detections:
    """
    <summary>한 프레임의 탐지 결과를 numpy 배열로 담은 묶음.</summary>
    <param name="boxes">(N, 4) int 배열. 각 줄이 x1, y1, x2, y2.</param>
    <param name="confs">(N,) float 배열. 신뢰도.</param>
    <param name="classes">(N,) int 배열. 클래스 번호.</param>
    <input>Detector.predict()가 만든다.</input>
    <output>FrameProcessor가 그리기와 집계에 쓴다.</output>
    <reason>
    ultralytics Results 객체를 그대로 들고 있으면 GPU 텐서가 남아 있을 수 있다.
    추론하지 않는 프레임에서 직전 결과를 재사용하려면 가벼운 numpy 배열로 꺼내 두는 편이 안전하고 빠르다.
    </reason>
    """

    boxes: np.ndarray
    confs: np.ndarray
    classes: np.ndarray

    def __len__(self):
        """
        <summary>탐지된 물체 수를 돌려준다.</summary>
        <input>없음.</input>
        <output>정수.</output>
        <reason>len(detections)로 바로 개수를 쓸 수 있게 한다.</reason>
        """
        return len(self.classes)


def precision_options(half):
    """
    <summary>FP16(half) 사용 여부를 predict()에 넘길 인자로 바꾼다.</summary>
    <param name="half">True면 16비트 실수(FP16)로 계산한다.</param>
    <input>FP16 여부.</input>
    <output>predict()에 덧붙일 인자 딕셔너리. FP32면 빈 딕셔너리.</output>
    <reason>
    ultralytics 8.4부터 half=True는 quantize=16이라는 새 이름으로 바뀌었고, 옛 이름을 쓰면 매번 경고가 뜬다.
    설치된 버전에 맞는 이름을 골라 넘기면 예전 버전에서도 같은 코드가 동작한다.
    </reason>
    """
    if not half:
        return {}

    if SUPPORTS_QUANTIZE:
        return {"quantize": 16}

    return {"half": True}


class Detector:
    """
    <summary>YOLO 모델을 불러와 프레임 한 장을 추론하고, 결과를 Detections로 돌려준다.</summary>
    <param name="settings">RunSettings 객체.</param>
    <input>load()로 모델을 준비하고 predict(frame)으로 추론한다.</input>
    <output>predict()가 (Detections, 추론 시간 ms, 전체 시간 ms) 튜플을 돌려준다.</output>
    <reason>모델 준비(불러오기, 워밍업, FP16 확인)와 추론을 한곳에 모아, 창 모드와 웹 모드가 같은 코드를 쓰게 한다.</reason>
    """

    def __init__(self, settings):
        """
        <summary>설정을 받아 둔다. 모델은 아직 불러오지 않는다.</summary>
        <param name="settings">RunSettings 객체.</param>
        <input>실행 설정.</input>
        <output>없음.</output>
        <reason>불러오기 실패를 load()의 반환값으로 알려 주기 위해 생성자에서는 무거운 일을 하지 않는다.</reason>
        """
        self.settings = settings
        self.model = None
        self.names = {}
        self.class_ids = None

    def load(self):
        """
        <summary>모델 파일(또는 OpenVINO 폴더)을 불러온다.</summary>
        <input>없음. settings.model_path를 쓴다.</input>
        <output>성공하면 True, 실패하면 False.</output>
        <reason>
        OpenVINO 폴더는 이름만으로 작업 종류를 알 수 없어 task="detect"를 함께 알려 준다.
        최신 모델을 못 열면 무엇으로 바꾸면 되는지(yolo11n.pt) 바로 안내한다.
        </reason>
        """
        path = self.settings.model_path
        print(f"[정보] 모델을 불러옵니다: {path}")

        try:
            self.model = YOLO(path, task="detect")
        except Exception as error:
            print(f"[오류] 모델을 불러오지 못했습니다: {error}")
            print("다음을 확인하세요.")
            print("  1. 인터넷에 연결되어 있는가? (첫 실행 때 모델 파일을 내려받는다)")
            print("  2. ultralytics가 최신인가?  pip install -U ultralytics")
            print(f"  3. 그래도 안 되면 --model {config.FALLBACK_MODEL_PATH} 로 실행한다.")

            if self.settings.openvino:
                print("  4. OpenVINO 폴더가 깨졌다면 --openvino 없이 실행하거나, export를 다시 한다.")

            return False

        self.names = self.model.names
        return True

    def predict(self, frame, half=None):
        """
        <summary>프레임 한 장을 추론한다.</summary>
        <param name="frame">BGR 이미지(numpy 배열).</param>
        <param name="half">FP16 여부. None이면 설정값을 쓴다.</param>
        <input>BGR 프레임.</input>
        <output>(Detections, 신경망 추론 시간 ms, predict() 전체 시간 ms) 튜플.</output>
        <reason>
        classes 인자로 필터를 넘기면 원하는 클래스만 남긴 결과를 받는다.
        전체 시간(전처리+추론+후처리)을 따로 재는 이유는 FPS에 영향을 주는 것이 신경망 계산만이 아니기 때문이다.
        </reason>
        """
        if half is None:
            half = self.settings.half

        start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            imgsz=self.settings.imgsz,
            conf=self.settings.conf,
            device=self.settings.device,
            classes=self.class_ids,
            verbose=False,
            **precision_options(half),
        )

        total_ms = (time.perf_counter() - start) * 1000
        result = results[0]
        boxes = result.boxes

        detections = Detections(
            boxes=boxes.xyxy.cpu().numpy().astype(int),
            confs=boxes.conf.cpu().numpy(),
            classes=boxes.cls.cpu().numpy().astype(int),
        )

        return detections, result.speed["inference"], total_ms

    def warm_up(self, frame_shape):
        """
        <summary>카메라 프레임과 같은 크기의 빈 이미지로 추론을 몇 번 미리 돌린다.</summary>
        <param name="frame_shape">카메라 프레임 모양 (세로, 가로, 3).</param>
        <input>프레임 모양.</input>
        <output>첫 추론에 걸린 시간(ms).</output>
        <reason>
        첫 추론은 모델을 장치에 올리고 계산 방법을 고르느라 몇 초씩 걸린다.
        카메라와 같은 크기로 미리 돌려 두면 실제 영상의 첫 몇 프레임이 멈춘 듯 보이지 않는다.
        </reason>
        """
        dummy = np.zeros(frame_shape, dtype=np.uint8)
        _, _, first_ms = self.predict(dummy)

        for _ in range(WARMUP_RUNS - 1):
            self.predict(dummy)

        return first_ms

    def verify_half(self, frame_shape):
        """
        <summary>FP16 결과가 FP32와 같은지 예제 사진으로 확인하고, 다르면 FP16을 끈다.</summary>
        <param name="frame_shape">카메라 프레임 모양. 예제 사진을 같은 크기로 맞춰 확인한다.</param>
        <input>프레임 모양.</input>
        <output>FP16을 그대로 쓰면 True, 껐으면 False.</output>
        <reason>
        일부 GPU(특히 GTX 16 시리즈)는 특정 입력 크기에서 FP16 결과가 통째로 사라진다(STEP 6에서 확인).
        오류 없이 "0개 탐지"로만 나타나 알아채기 어려우므로, 실시간 영상을 시작하기 전에 확인한다.
        물체가 확실히 있는 예제 사진(bus.jpg)을 카메라와 같은 크기로 바꿔 써야 실제와 같은 조건이 된다.
        </reason>
        """
        sample = cv2.imread(str(ASSETS / "bus.jpg"))

        if sample is None:
            return True

        height, width = frame_shape[:2]
        sample = cv2.resize(sample, (width, height))

        for _ in range(WARMUP_RUNS):
            fp32, _, _ = self.predict(sample, half=False)

        for _ in range(WARMUP_RUNS):
            fp16, _, _ = self.predict(sample, half=True)

        print(f"[정보] FP16 확인: FP32 {len(fp32)}개 / FP16 {len(fp16)}개 탐지")

        if len(fp32) == len(fp16):
            return True

        print("[안내] 이 GPU에서는 FP16 결과가 FP32와 달라 FP32로 실행합니다.")
        self.settings.half = False
        self.settings.notes["half"] = "FP16 결과 불일치 → FP32"
        return False

    def prepare(self, frame_shape, class_names):
        """
        <summary>클래스 필터 설정, FP16 확인, 워밍업을 차례로 한다.</summary>
        <param name="frame_shape">카메라 프레임 모양.</param>
        <param name="class_names">--classes로 받은 이름 목록. 없으면 None.</param>
        <input>프레임 모양과 클래스 이름 목록.</input>
        <output>성공하면 True, 클래스 이름이 잘못되었으면 False.</output>
        <reason>잘못된 클래스 이름은 실시간 영상을 시작하기 전에 알려 줘야 학생이 바로 고칠 수 있다.</reason>
        """
        class_ids, unknown = resolve_class_filter(class_names, self.names)

        if unknown:
            print(f"[오류] 모르는 클래스 이름: {', '.join(unknown)}")
            print("사용할 수 있는 이름(일부):")
            print("  " + ", ".join(list(self.names.values())[:40]) + ", ...")
            print("띄어쓰기가 있는 이름은 cell_phone처럼 밑줄로 적으세요.")
            return False

        self.class_ids = class_ids

        if self.settings.half:
            self.verify_half(frame_shape)

        print("[정보] 워밍업 중...")
        first_ms = self.warm_up(frame_shape)
        print(f"[정보] 워밍업 완료 (첫 추론 {first_ms:.0f} ms, 처음 {WARMUP_RUNS}회는 측정에서 제외)")

        return True


# ============================================================
# 3. 그리기 (STEP 5·6과 같은 cv2.rectangle / cv2.putText 방식)
# ============================================================

def color_for_class(class_id):
    """
    <summary>클래스 번호에 해당하는 색을 고른다.</summary>
    <param name="class_id">YOLO 클래스 번호.</param>
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


def draw_detections(image, detections, names):
    """
    <summary>탐지 상자와 "이름 신뢰도" 라벨을 그린다.</summary>
    <param name="image">BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="detections">Detections 객체.</param>
    <param name="names">클래스 번호 → 이름 딕셔너리.</param>
    <input>이미지, 탐지 결과, 이름표.</input>
    <output>없음. image가 직접 바뀐다.</output>
    <reason>
    STEP 6과 같은 방법이다. 실시간에서는 프레임마다 복사하는 비용도 아끼려고 원본 위에 바로 그린다.
    상자가 화면 맨 위에 붙으면 라벨이 잘리므로 그때는 상자 안쪽에 그린다.
    </reason>
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.boxes[i]
        class_id = int(detections.classes[i])
        color = color_for_class(class_id)
        label = f"{names[class_id]} {detections.confs[i]:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_height = text_height + baseline + 6

        if y1 - label_height < 0:
            top = y1
        else:
            top = y1 - label_height

        cv2.rectangle(image, (x1, top), (x1 + text_width + 8, top + label_height), color, -1)
        cv2.putText(image, label, (x1 + 4, top + text_height + 3), font, font_scale, text_color_for(color), thickness, cv2.LINE_AA)


def draw_info_panel(image, lines, anchor="left"):
    """
    <summary>반투명 배경 상자 위에 여러 줄의 글자를 쓴다. 왼쪽 위 또는 오른쪽 위에 놓는다.</summary>
    <param name="image">BGR 이미지. 이 배열이 직접 바뀐다.</param>
    <param name="lines">(문자열, 글자 BGR 색) 또는 (문자열, 글자 색, 견본 색) 튜플 목록. 견본 색이 있으면 글자 앞에 작은 색 사각형을 그린다.</param>
    <param name="anchor">"left"면 왼쪽 위, "right"면 오른쪽 위.</param>
    <input>이미지, 줄 목록, 위치.</input>
    <output>없음. image가 직접 바뀐다.</output>
    <reason>
    STEP 5의 draw_info_panel과 같다. getTextSize로 가장 긴 줄을 재서 배경 상자 폭을 맞춘다.
    파란색 같은 어두운 클래스 색은 검은 배경 위 글자로 읽기 어려워, 글자는 밝게 두고 색은 견본 사각형으로 보여 준다.
    </reason>
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 24
    padding = 10
    margin = 10
    swatch_size = 10
    swatch_gap = 8

    has_swatch = any(len(line) == 3 for line in lines)

    if has_swatch:
        text_offset = swatch_size + swatch_gap
    else:
        text_offset = 0

    box_width = 0

    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line[0], font, font_scale, thickness)
        box_width = max(box_width, text_width)

    box_width += padding * 2 + text_offset
    box_height = line_height * len(lines) + padding

    if anchor == "right":
        x = image.shape[1] - box_width - margin
    else:
        x = margin

    y = margin

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, dst=image)

    for i, line in enumerate(lines):
        text = line[0]
        color = line[1]
        text_y = y + padding + line_height * (i + 1) - 8
        text_x = x + padding + text_offset

        if len(line) == 3:
            swatch_top = text_y - swatch_size
            cv2.rectangle(image, (x + padding, swatch_top), (x + padding + swatch_size, text_y), line[2], -1)

        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


# ============================================================
# 4. 프레임 처리 (skip, 직전 결과 재사용, FPS, 화면 정보)
# ============================================================

class FpsMeter:
    """
    <summary>최근 N프레임의 시각으로 이동평균 FPS를 계산한다.</summary>
    <param name="size">평균 낼 최근 프레임 수.</param>
    <input>프레임마다 tick()을 호출한다.</input>
    <output>fps()가 이동평균 FPS를 돌려준다.</output>
    <reason>STEP 5와 같은 방식이다. deque(maxlen=N)가 오래된 시각을 자동으로 밀어낸다.</reason>
    """

    def __init__(self, size):
        """
        <summary>프레임 시각을 담을 상자를 만든다.</summary>
        <param name="size">평균 낼 최근 프레임 수.</param>
        <input>정수.</input>
        <output>없음.</output>
        <reason>maxlen을 정하면 가장 오래된 값이 저절로 빠진다.</reason>
        """
        self.times = deque(maxlen=size)

    def tick(self):
        """
        <summary>지금 시각을 기록한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>time.perf_counter()는 짧은 간격을 재기에 가장 정밀한 시계다.</reason>
        """
        self.times.append(time.perf_counter())

    def fps(self):
        """
        <summary>최근 프레임들의 평균 FPS를 계산한다.</summary>
        <input>없음.</input>
        <output>FPS(float). 기록이 2개 미만이면 0.0.</output>
        <reason>(프레임 수 - 1) / (마지막 시각 - 첫 시각)이 최근 구간의 1초당 처리 장수다.</reason>
        """
        if len(self.times) < 2:
            return 0.0

        elapsed = self.times[-1] - self.times[0]

        if elapsed <= 0:
            return 0.0

        return (len(self.times) - 1) / elapsed


class FrameProcessor:
    """
    <summary>프레임마다 skip 규칙에 따라 추론하거나 직전 결과를 재사용하고, 상자와 화면 정보를 그린다.</summary>
    <param name="detector">준비된 Detector 객체.</param>
    <param name="settings">RunSettings 객체.</param>
    <input>process(frame)에 카메라 프레임을 넣는다.</input>
    <output>process()가 (그림이 그려진 프레임, 통계 딕셔너리) 튜플을 돌려준다.</output>
    <reason>
    skip=2면 짝수 번째 프레임만 추론하고, 홀수 번째 프레임에는 직전 상자를 그대로 그린다.
    사람이 움직이는 속도에 비해 1/30초 전 상자는 거의 같은 자리라, 눈에 띄는 차이 없이 추론 횟수를 절반으로 줄인다.
    창 모드와 웹 모드가 이 클래스를 함께 써서 동작이 똑같이 유지된다.
    </reason>
    """

    def __init__(self, detector, settings):
        """
        <summary>프레임 번호, 직전 결과, FPS 측정기를 준비한다.</summary>
        <param name="detector">준비된 Detector 객체.</param>
        <param name="settings">RunSettings 객체.</param>
        <input>탐지기와 설정.</input>
        <output>없음.</output>
        <reason>직전 결과를 들고 있어야 추론하지 않는 프레임에서도 상자를 그릴 수 있다.</reason>
        """
        self.detector = detector
        self.settings = settings
        self.frame_index = 0
        self.last_detections = None
        self.last_inference_ms = 0.0
        self.last_total_ms = 0.0
        self.fps_meter = FpsMeter(FPS_WINDOW)

    def process(self, frame):
        """
        <summary>프레임 한 장을 처리한다.</summary>
        <param name="frame">BGR 프레임. 이 배열 위에 바로 그린다.</param>
        <input>BGR 프레임.</input>
        <output>(그림이 그려진 프레임, 통계 딕셔너리) 튜플.</output>
        <reason>
        frame_index % skip == 0일 때만 추론한다. 첫 프레임은 직전 결과가 없으므로 반드시 추론한다.
        통계 딕셔너리는 웹 모드의 /stats가 그대로 JSON으로 내보낼 수 있게 기본 자료형만 담는다.
        </reason>
        """
        should_infer = self.last_detections is None or self.frame_index % self.settings.skip == 0
        self.frame_index += 1

        if should_infer:
            detections, inference_ms, total_ms = self.detector.predict(frame)
            self.last_detections = detections
            self.last_inference_ms = inference_ms
            self.last_total_ms = total_ms
        else:
            detections = self.last_detections

        draw_detections(frame, detections, self.detector.names)

        counts = Counter(self.detector.names[int(c)] for c in detections.classes)
        self.fps_meter.tick()
        fps = self.fps_meter.fps()

        if should_infer:
            boxes_state = "new"
        else:
            boxes_state = "reused"

        info_lines = [
            (f"FPS: {fps:.1f}", (120, 230, 120)),
            (f"Inference: {self.last_inference_ms:.0f} ms (total {self.last_total_ms:.0f})", (120, 230, 120)),
            (f"Objects: {len(detections)}  [{boxes_state}]", (255, 255, 255)),
            (f"Device: {self.settings.device_label()}", (255, 255, 255)),
            (f"imgsz {self.settings.imgsz}  skip {self.settings.skip}  conf {self.settings.conf}", (200, 200, 200)),
        ]
        draw_info_panel(frame, info_lines, anchor="left")

        count_lines = [("Counts", (255, 255, 255))]

        if counts:
            for name, number in counts.most_common():
                count_lines.append((f"{name}: {number}", (235, 235, 235), color_for_class(self._class_id(name))))
        else:
            count_lines.append(("none", (170, 170, 170)))

        draw_info_panel(frame, count_lines, anchor="right")

        stats = {
            "fps": round(fps, 1),
            "inference_ms": round(self.last_inference_ms, 1),
            "total_ms": round(self.last_total_ms, 1),
            "total": len(detections),
            "counts": dict(counts.most_common()),
            "device": self.settings.device_label(),
            "imgsz": self.settings.imgsz,
            "skip": self.settings.skip,
            "conf": self.settings.conf,
        }

        return frame, stats

    def _class_id(self, name):
        """
        <summary>클래스 이름으로 클래스 번호를 찾는다.</summary>
        <param name="name">클래스 이름.</param>
        <input>문자열.</input>
        <output>클래스 번호(int). 못 찾으면 0.</output>
        <reason>집계 표의 글자색을 상자 색과 맞추기 위해 이름에서 번호를 되찾는다.</reason>
        """
        for class_id, class_name in self.detector.names.items():
            if class_name == name:
                return class_id

        return 0


# ============================================================
# 5. 카메라와 저장소
# ============================================================

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
    print("  1. 줌, 팀즈, 카메라 앱, 다른 STEP 서버 등이 카메라를 쓰고 있지 않은가?")
    print("  2. Windows 설정 > 개인 정보 및 보안 > 카메라 에서 '데스크톱 앱의 카메라 액세스'가 켜져 있는가?")
    print("  3. 노트북의 카메라 차단 스위치나 Fn 단축키로 카메라가 꺼져 있지 않은가?")
    print("  4. 외장 카메라라면 config.py의 CAMERA_INDEX를 1, 2로 바꿔 보았는가?")


def open_camera(index, width, height):
    """
    <summary>지정한 번호의 웹캠을 열고 해상도를 설정한다.</summary>
    <param name="index">카메라 번호.</param>
    <param name="width">요청할 가로 해상도.</param>
    <param name="height">요청할 세로 해상도.</param>
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


class Camera:
    """
    <summary>웹캠을 한 번만 열고, 별도 스레드에서 계속 읽어 최신 프레임 1장만 보관한다. (웹 모드용)</summary>
    <param name="index">카메라 번호.</param>
    <param name="width">요청할 가로 해상도.</param>
    <param name="height">요청할 세로 해상도.</param>
    <input>start()로 읽기를 시작하고, get_frame()으로 최신 프레임을 가져간다.</input>
    <output>get_frame()이 (프레임 번호, BGR 이미지) 튜플을 돌려준다.</output>
    <reason>
    요청마다 카메라를 열면 장치 충돌이 나기 때문이다(STEP 2).
    탐지 스레드, 원본 스트림, 촬영 요청이 모두 이 한 카메라의 최신 프레임을 나눠 쓴다.
    </reason>
    """

    def __init__(self, index, width, height):
        """
        <summary>카메라 설정과 스레드에서 쓸 값을 준비한다.</summary>
        <param name="index">카메라 번호.</param>
        <param name="width">요청할 가로 해상도.</param>
        <param name="height">요청할 세로 해상도.</param>
        <input>카메라 번호와 해상도.</input>
        <output>없음.</output>
        <reason>lock은 읽기 스레드와 다른 스레드가 같은 프레임을 동시에 건드리지 않게 막는다.</reason>
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
        <reason>daemon 스레드로 만들면 메인 프로그램이 끝날 때 스레드가 프로그램을 붙잡지 않는다.</reason>
        """
        self.cap = open_camera(self.index, self.width, self.height)

        if self.cap is None:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        return True

    def _read_loop(self):
        """
        <summary>스레드 안에서 카메라 프레임을 계속 읽어 최신 프레임을 덮어쓴다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>연속으로 실패했을 때만 카메라가 끊긴 것으로 보고 멈춘다.</reason>
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
        <reason>탐지 스레드가 복사본 위에 상자를 그려도 원본 스트림(/video_feed)에는 영향이 없다.</reason>
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
    <summary>탐지 결과 프레임을 captures/detected 폴더에 저장하고, 저장된 목록을 관리한다.</summary>
    <param name="directory">저장 폴더 경로.</param>
    <input>save()에 BGR 프레임을 넘기고, recent()로 최근 목록을 받는다.</input>
    <output>save()는 저장한 파일 이름, recent()는 파일 이름 목록을 돌려준다.</output>
    <reason>
    STEP 3과 같은 이름 규칙(YYYYMMDD_HHMMSS.jpg)을 쓰되 폴더를 나눈다.
    상자가 그려진 사진이 원본 캡처 폴더에 섞이면 STEP 4·6에서 그 위에 다시 YOLO를 돌리게 되기 때문이다.
    </reason>
    """

    def __init__(self, directory):
        """
        <summary>저장 폴더를 만들고, 동시 저장을 막을 lock을 준비한다.</summary>
        <param name="directory">저장 폴더 경로.</param>
        <input>폴더 경로.</input>
        <output>없음.</output>
        <reason>이름 고르기와 저장을 lock 안에서 한 번에 처리해 빠른 연속 촬영에도 덮어쓰지 않는다.</reason>
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def save(self, frame):
        """
        <summary>프레임을 jpg 파일로 저장한다.</summary>
        <param name="frame">저장할 BGR 이미지.</param>
        <input>BGR 이미지.</input>
        <output>저장한 파일 이름. 실패하면 None.</output>
        <reason>같은 초에 여러 장을 찍으면 _2, _3을 붙여 이름이 겹치지 않게 한다.</reason>
        """
        with self.lock:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stamp}.jpg"
            number = 2

            while (self.directory / filename).exists():
                filename = f"{stamp}_{number}.jpg"
                number += 1

            path = self.directory / filename
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not ok:
            print(f"[오류] 이미지를 저장하지 못했습니다: {path}")
            return None

        print(f"[정보] 탐지 결과 저장: {path}")
        return filename

    def recent(self, limit):
        """
        <summary>저장된 파일 이름을 최신순으로 돌려준다.</summary>
        <param name="limit">돌려줄 최대 개수.</param>
        <input>최대 개수.</input>
        <output>파일 이름 목록. 가장 최근 것이 맨 앞이다.</output>
        <reason>이름 규칙에 맞는 파일만 골라, 폴더에 다른 파일이 있어도 목록에 섞이지 않게 한다.</reason>
        """
        names = [p.name for p in self.directory.glob("*.jpg") if is_valid_capture_name(p.name)]
        names.sort(key=capture_sort_key, reverse=True)

        return names[:limit]


def is_valid_capture_name(filename):
    """
    <summary>파일 이름이 저장 이름 규칙(YYYYMMDD_HHMMSS.jpg)에 맞는지 확인한다.</summary>
    <param name="filename">확인할 파일 이름.</param>
    <input>문자열.</input>
    <output>맞으면 True.</output>
    <reason>허용할 모양만 통과시키는 화이트리스트 방식이라, ".." 같은 경로 조작을 빠짐없이 막는다(STEP 3).</reason>
    """
    return CAPTURE_NAME_PATTERN.match(filename) is not None


def capture_sort_key(filename):
    """
    <summary>저장 파일 이름을 촬영 순서대로 정렬하기 위한 기준값을 만든다.</summary>
    <param name="filename">규칙에 맞는 파일 이름.</param>
    <input>문자열.</input>
    <output>(시각 문자열, 같은 초 안의 번호) 튜플.</output>
    <reason>"_10"이 "_2"보다 앞에 오지 않도록 번호는 숫자로 비교한다.</reason>
    """
    parts = filename[:-4].split("_")
    stamp = parts[0] + parts[1]

    if len(parts) == 3:
        number = int(parts[2])
    else:
        number = 1

    return stamp, number


# ============================================================
# 6. (a) 창 모드
# ============================================================

def run_window(settings, detector, store):
    """
    <summary>cv2.imshow 창에 실시간 탐지 결과를 보여 준다. s: 저장, q: 종료.</summary>
    <param name="settings">RunSettings 객체.</param>
    <param name="detector">준비된 Detector 객체.</param>
    <param name="store">CaptureStore 객체.</param>
    <input>설정, 탐지기, 저장소. 키보드 입력.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 카메라 오류는 1.</output>
    <reason>창 모드는 스레드 없이 "읽기 → 처리 → 표시"를 한 줄로 반복하므로, 처리 시간이 곧바로 FPS에 드러난다.</reason>
    """
    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if cap is None:
        print_camera_help(config.CAMERA_INDEX)
        return 1

    processor = FrameProcessor(detector, settings)

    print("[정보] 카메라가 열렸습니다.")
    print(f"[조작] 창을 클릭한 뒤  s: 탐지 결과 저장  {config.QUIT_KEY}: 종료")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
                print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            annotated, _ = processor.process(frame)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("[정보] 종료 키를 눌러 프로그램을 끝냅니다.")
                break

            if key == ord("s"):
                store.save(annotated)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[정보] 창이 닫혀 프로그램을 끝냅니다.")
                break

    except KeyboardInterrupt:
        print("[정보] Ctrl+C로 프로그램을 끝냅니다.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[정보] 카메라와 창을 해제했습니다.")

    return 0


# ============================================================
# 7. (b) 웹 모드
# ============================================================

class DetectionWorker:
    """
    <summary>별도 스레드에서 카메라 최신 프레임을 FrameProcessor로 처리하고, 결과 프레임·JPEG·통계를 보관한다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <param name="processor">FrameProcessor 객체.</param>
    <input>start()로 처리를 시작한다.</input>
    <output>read_jpeg(), read_frame(), read_stats()로 최신 결과를 꺼내 간다.</output>
    <reason>
    브라우저 접속마다 YOLO를 돌리면 접속 수만큼 느려진다.
    추론과 JPEG 변환은 이 스레드가 한 번만 하고, 모든 접속과 촬영 요청이 같은 결과를 나눠 쓴다.
    </reason>
    """

    def __init__(self, camera, processor):
        """
        <summary>처리 스레드가 쓸 값을 준비한다.</summary>
        <param name="camera">Camera 객체.</param>
        <param name="processor">FrameProcessor 객체.</param>
        <input>카메라와 처리기.</input>
        <output>없음.</output>
        <reason>error 값을 두어, 스레드 안에서 생긴 문제를 웹 페이지에도 보여 줄 수 있게 한다.</reason>
        """
        self.camera = camera
        self.processor = processor
        self.lock = threading.Lock()
        self.frame = None
        self.jpeg = None
        self.result_id = 0
        self.stats = {"error": ""}
        self.running = False
        self.thread = None

    def start(self):
        """
        <summary>처리 스레드를 시작한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>daemon 스레드로 만들면 메인 프로그램이 끝날 때 함께 정리된다.</reason>
        """
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_error(self, message):
        """
        <summary>오류를 콘솔과 통계에 남기고 처리 스레드를 멈춘다.</summary>
        <param name="message">오류 문장.</param>
        <input>문자열.</input>
        <output>없음.</output>
        <reason>스레드 안의 오류는 화면에 드러나지 않으므로 콘솔과 웹 양쪽에 남긴다.</reason>
        """
        print(f"[오류] {message}")

        with self.lock:
            self.stats["error"] = message

        self.running = False

    def _loop(self):
        """
        <summary>새 카메라 프레임이 오면 처리하고, 결과를 JPEG로 바꿔 보관한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>이미 처리한 프레임 번호면 건너뛴다. 같은 프레임을 두 번 세면 skip 규칙과 FPS가 틀어진다.</reason>
        """
        last_frame_id = 0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]

        while self.running:
            if not self.camera.running:
                self._set_error("카메라 연결이 끊겼습니다. 서버를 다시 시작하세요.")
                break

            frame_id, frame = self.camera.get_frame()

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = frame_id

            try:
                annotated, stats = self.processor.process(frame)
            except Exception as error:
                self._set_error(f"YOLO 추론 중 문제가 생겼습니다: {error}")
                break

            ok, buffer = cv2.imencode(".jpg", annotated, encode_params)

            if not ok:
                continue

            stats["error"] = ""

            with self.lock:
                self.frame = annotated
                self.jpeg = buffer.tobytes()
                self.result_id += 1
                self.stats = stats

    def read_jpeg(self):
        """
        <summary>최신 결과 JPEG와 결과 번호를 돌려준다.</summary>
        <input>없음.</input>
        <output>(결과 번호, JPEG bytes) 튜플. 아직 없으면 (0, None).</output>
        <reason>bytes는 바뀌지 않는 값이라 복사하지 않고 넘겨도 안전하다.</reason>
        """
        with self.lock:
            return self.result_id, self.jpeg

    def read_frame(self):
        """
        <summary>최신 결과 프레임(상자가 그려진 BGR 배열)의 복사본을 돌려준다.</summary>
        <input>없음.</input>
        <output>BGR 이미지. 아직 없으면 None.</output>
        <reason>촬영 시 JPEG를 다시 풀지 않고 원래 배열을 고화질로 저장하기 위해서다.</reason>
        """
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def read_stats(self):
        """
        <summary>최신 통계의 복사본을 돌려준다.</summary>
        <input>없음.</input>
        <output>통계 딕셔너리.</output>
        <reason>JSON으로 바꾸는 도중에 처리 스레드가 값을 바꿔도 안전하도록 복사본을 준다.</reason>
        """
        with self.lock:
            snapshot = dict(self.stats)

        if "counts" in snapshot:
            snapshot["counts"] = dict(snapshot["counts"])

        return snapshot

    def stop(self):
        """
        <summary>처리 스레드를 멈춘다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>카메라를 해제하기 전에 처리 스레드부터 멈춰야 해제된 카메라를 읽으려 하지 않는다.</reason>
        """
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None


def generate_raw_mjpeg(camera):
    """
    <summary>원본 카메라 프레임을 MJPEG로 내보낸다. (STEP 3의 /video_feed)</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <input>Camera 객체.</input>
    <output>MJPEG 조각(bytes) 제너레이터.</output>
    <reason>탐지 스트림과 나란히 열어 보면 skip과 추론 지연이 원본보다 얼마나 늦는지 비교할 수 있다.</reason>
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


def generate_detect_mjpeg(worker):
    """
    <summary>탐지 결과 JPEG를 MJPEG로 내보낸다. (/detect_feed)</summary>
    <param name="worker">실행 중인 DetectionWorker 객체.</param>
    <input>DetectionWorker 객체.</input>
    <output>MJPEG 조각(bytes) 제너레이터.</output>
    <reason>새 결과가 나왔을 때만 보내므로, 처리가 느린 CPU에서도 같은 그림을 반복해 보내지 않는다.</reason>
    """
    last_id = 0

    while worker.running:
        result_id, jpeg = worker.read_jpeg()

        if jpeg is None or result_id == last_id:
            time.sleep(0.01)
            continue

        last_id = result_id

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def create_app(camera, worker, store):
    """
    <summary>STEP 3 서버에 /detect_feed와 /stats를 더한 Flask 앱을 만든다.</summary>
    <param name="camera">Camera 객체.</param>
    <param name="worker">DetectionWorker 객체.</param>
    <param name="store">CaptureStore 객체.</param>
    <input>카메라, 처리 스레드, 저장소.</input>
    <output>Flask 앱 객체.</output>
    <reason>STEP 3의 주소(/, /video_feed, /capture, /captures)는 그대로 두고, 촬영만 "탐지 결과 프레임 저장"으로 바꾼다.</reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>templates/step7_realtime.html을 그려서 돌려준다.</summary>
        <input>GET / 요청.</input>
        <output>HTML 응답.</output>
        <reason>STEP 2·3의 템플릿은 그대로 두고 7단계용 페이지를 따로 둔다.</reason>
        """
        return render_template("step7_realtime.html", history_limit=HISTORY_LIMIT)

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>원본 카메라 MJPEG 스트림.</summary>
        <input>GET /video_feed 요청.</input>
        <output>multipart/x-mixed-replace 스트리밍 응답.</output>
        <reason>STEP 3과 같은 주소를 남겨, 원본과 탐지 결과를 비교할 수 있게 한다.</reason>
        """
        return Response(generate_raw_mjpeg(camera), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/detect_feed")
    def detect_feed():
        """
        <summary>탐지 결과가 그려진 MJPEG 스트림.</summary>
        <input>GET /detect_feed 요청.</input>
        <output>multipart/x-mixed-replace 스트리밍 응답.</output>
        <reason>원본 스트림과 같은 형식이라 페이지에서는 img 태그의 주소만 바꾸면 된다.</reason>
        """
        return Response(generate_detect_mjpeg(worker), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        """
        <summary>FPS, 추론 시간, 클래스별 개수, 설정을 JSON으로 돌려준다.</summary>
        <input>GET /stats 요청.</input>
        <output>통계 JSON.</output>
        <reason>숫자는 영상 위 글자보다 HTML 표로 보여 주는 편이 읽기 쉽고, 다른 프로그램도 가져다 쓸 수 있다.</reason>
        """
        return jsonify(worker.read_stats())

    @app.route("/capture", methods=["POST"])
    def capture():
        """
        <summary>현재 탐지 결과 프레임(상자 포함)을 저장하고 파일 이름과 그 순간의 개수를 JSON으로 돌려준다.</summary>
        <input>POST /capture 요청.</input>
        <output>성공: {"filename", "url", "counts"} (200). 실패: {"error"} (503 또는 500).</output>
        <reason>서버에 파일을 만드는 동작이므로 GET이 아닌 POST로 받는다(STEP 3).</reason>
        """
        frame = worker.read_frame()

        if frame is None or not worker.running:
            return jsonify(error="탐지 결과가 아직 없어 저장할 수 없습니다. 서버 콘솔을 확인하세요."), 503

        filename = store.save(frame)

        if filename is None:
            return jsonify(error="이미지를 저장하지 못했습니다. 폴더 권한을 확인하세요."), 500

        counts = worker.read_stats().get("counts", {})
        return jsonify(filename=filename, url=f"/captures/{filename}", counts=counts)

    @app.route("/captures")
    def capture_list():
        """
        <summary>최근 저장 이력을 JSON으로 돌려준다.</summary>
        <input>GET /captures 요청.</input>
        <output>{"captures": [{"filename", "url"}, ...]} 최신순.</output>
        <reason>페이지를 새로 고쳐도 저장 이력이 그대로 보이게 한다.</reason>
        """
        items = [{"filename": name, "url": f"/captures/{name}"} for name in store.recent(HISTORY_LIMIT)]

        return jsonify(captures=items)

    @app.route("/captures/<filename>")
    def capture_file(filename):
        """
        <summary>저장된 탐지 결과 이미지를 돌려준다.</summary>
        <param name="filename">주소에 들어온 파일 이름.</param>
        <input>GET /captures/파일이름 요청.</input>
        <output>jpg 응답. 이름 규칙에 맞지 않거나 없으면 404.</output>
        <reason>이름 규칙 검사와 send_from_directory로 경로 조작(../)을 두 겹으로 막는다(STEP 3).</reason>
        """
        if not is_valid_capture_name(filename):
            abort(404)

        return send_from_directory(store.directory, filename, mimetype="image/jpeg")

    return app


def run_web(settings, detector, store):
    """
    <summary>카메라 스레드와 탐지 스레드를 시작하고 Flask 서버를 실행한다.</summary>
    <param name="settings">RunSettings 객체.</param>
    <param name="detector">준비된 Detector 객체.</param>
    <param name="store">CaptureStore 객체.</param>
    <input>설정, 탐지기, 저장소.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    종료는 시작의 반대 순서(탐지 스레드 → 카메라)로 한다. atexit에 등록해 두면 Ctrl+C로 끝내도 빠짐없이 정리된다.
    debug와 use_reloader를 끄는 이유는 Flask가 프로그램을 두 번 실행해 카메라를 두 번 열지 않게 하기 위해서다.
    </reason>
    """
    camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if not camera.start():
        print_camera_help(config.CAMERA_INDEX)
        return 1

    worker = DetectionWorker(camera, FrameProcessor(detector, settings))

    atexit.register(camera.release)
    atexit.register(worker.stop)

    if not camera.wait_first_frame(timeout=5):
        print("[오류] 카메라는 열렸지만 5초 동안 영상이 들어오지 않았습니다.")
        print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
        return 1

    worker.start()
    app = create_app(camera, worker, store)

    print(f"[정보] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print("[정보] 서버를 끄려면 이 콘솔에서 Ctrl+C를 누르세요.")

    try:
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)

    except KeyboardInterrupt:
        print("[정보] Ctrl+C로 서버를 끝냅니다.")

    except OSError as error:
        print(f"[오류] 서버를 시작하지 못했습니다: {error}")
        print(f"{PORT}번 포트를 다른 STEP 서버가 쓰고 있지 않은지 확인하세요.")
        return 1

    return 0


# ============================================================
# 8. 시작
# ============================================================

def run(argv=None):
    """
    <summary>옵션 해석 → 설정 결정 → 모델 준비 → 선택한 모드 실행 순서로 동작한다.</summary>
    <param name="argv">명령줄 옵션 목록. None이면 실제 명령줄을 쓴다.</param>
    <input>명령줄 옵션.</input>
    <output>프로그램 종료 코드.</output>
    <reason>모델 준비(다운로드, FP16 확인, 워밍업)를 카메라보다 먼저 끝내, 카메라만 켜진 채 오래 기다리는 일이 없게 한다.</reason>
    """
    args = parse_args(argv)
    settings = resolve_settings(args)
    detector = Detector(settings)

    if not detector.load():
        return 1

    frame_shape = (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)

    if not detector.prepare(frame_shape, args.classes):
        return 1

    print_settings(settings, detector.class_ids, detector.names)

    try:
        store = CaptureStore(DETECTION_DIR)
    except OSError as error:
        print(f"[오류] 저장 폴더를 만들지 못했습니다: {error}")
        return 1

    if args.mode == "web":
        return run_web(settings, detector, store)

    return run_window(settings, detector, store)


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
#
# 2. 창 모드 (기본값: CPU, imgsz 320, skip 2)
#      python step7_realtime_yolo.py
#    창을 클릭한 뒤 s: 탐지 결과 저장(captures\detected), q: 종료
#
# 3. 웹 모드 (STEP 3 서버 + 탐지 스트림)
#      python step7_realtime_yolo.py --mode web
#    브라우저에서 http://127.0.0.1:5000 에 접속한다. "촬영" 버튼은 탐지 결과 프레임을 저장한다.
#    원본 스트림 /video_feed, 탐지 스트림 /detect_feed, 통계 /stats 를 각각 열어 볼 수도 있다.
#
# 4. 자주 쓰는 옵션
#      --device cuda:0             NVIDIA GPU (imgsz 640, skip 1, FP16 자동)
#      --device mps                Apple Silicon 맥
#      --imgsz 416 --skip 3        값을 직접 지정하면 자동 권장값보다 우선한다
#      --conf 0.6                  더 확실한 것만 표시
#      --classes person cup        사람과 컵만 탐지 (cell phone은 cell_phone으로)
#      --model yolo11n.pt          다른 모델로 실행
#      --openvino                  Intel CPU 가속 (아래 준비 필요)
#      -h                          모든 옵션 보기
#
# 5. OpenVINO 준비 (Intel CPU, 한 번만)
#      pip install openvino
#      yolo export model=yolo26n.pt format=openvino
#    → yolo26n_openvino_model 폴더가 생기면 --openvino로 실행한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 설정별 FPS를 표로 기록하자. 각 설정으로 30초쯤 실행하며 화면의 FPS와 Inference 값을 적는다.
#    카메라 앞 장면(사람 수, 밝기)은 되도록 같게 유지한다.
#
#    | 번호 | 명령 옵션                                   | 장치   | imgsz | skip | FPS | Inference(ms) | 탐지가 끊기는가? |
#    |------|---------------------------------------------|--------|-------|------|-----|---------------|------------------|
#    | A    | (옵션 없음)                                 | cpu    | 320   | 2    |     |               |                  |
#    | B    | --skip 1                                    | cpu    | 320   | 1    |     |               |                  |
#    | C    | --imgsz 640 --skip 1                        | cpu    | 640   | 1    |     |               |                  |
#    | D    | --imgsz 640 --skip 3                        | cpu    | 640   | 3    |     |               |                  |
#    | E    | --openvino                                  | cpu    | 320   | 2    |     |               |                  |
#    | F    | --device cuda:0                             | cuda:0 | 640   | 1    |     |               |                  |
#    | G    | --device cuda:0 --imgsz 320                 | cuda:0 | 320   | 1    |     |               |                  |
#
#    GPU가 없는 컴퓨터는 F, G를 건너뛰고, Intel CPU가 아니면 E를 건너뛴다.
#
# 2. 표를 보고 답해 보자.
#    - CPU vs GPU: A와 F의 FPS는 몇 배 차이 나는가? Inference(ms)는?
#    - imgsz: B와 C를 비교하자. 입력 칸 수는 4배인데 Inference도 4배가 되었는가?
#      작은 물체(멀리 있는 컵, 휴대폰)는 320에서 더 자주 놓치는가?
#    - skip: B와 A를 비교하자. skip 2는 추론을 절반만 하는데 FPS도 2배가 되었는가?
#      화면의 [new]/[reused] 표시를 보며, 손을 빠르게 움직일 때 상자가 따라오지 못하고 늦게 움직이는지 관찰하자.
#    - FPS가 카메라 속도(보통 15~30)에서 더 올라가지 않는 설정은 어느 것인가? 왜 그럴까? (STEP 5의 Capacity)
#
# 3. 클래스 필터
#    - --classes person 으로 실행하면 Inference(ms)가 줄어드는가?
#      (신경망은 80개 클래스를 모두 계산한 뒤 결과만 걸러 낸다. 그래서 속도는 거의 그대로다)
#    - 오른쪽 위 Counts 표의 숫자가 화면 속 사람 수와 맞는가?
#
# 4. 창 모드 vs 웹 모드
#    - 같은 설정에서 두 모드의 FPS를 비교하자. 웹 모드는 JPEG 변환과 전송이 더해지는데 FPS가 얼마나 떨어지는가?
#    - 웹 페이지에서 원본(/video_feed)과 탐지 결과(/detect_feed)를 나란히 열면 어느 쪽이 더 늦게 보이는가?
