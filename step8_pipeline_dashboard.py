"""
<summary>8단계: STEP 7 웹 모드를 확장해, 영상과 함께 "지금 프레임이 어느 단계를 지나는지"를 플로우차트와 로그로 실시간 보여 준다.</summary>
<reason>
실시간 인식이 느릴 때 "어디가 느린가"를 감으로 찾으면 엉뚱한 곳을 고치게 된다.
파이프라인을 capture → preprocess → inference → postprocess → visualize → output 여섯 단계로 나누고
단계마다 시간을 재서 화면에 띄우면, 병목이 어디인지 숫자로 보인다.
플로우차트의 설명 글은 이 파일의 단계 함수 docstring을 그대로 읽어 만든다. 코드와 설명이 따로 놀지 않는다.
</reason>
"""

import argparse
import atexit
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

import config
import step7_realtime_yolo as step7
from pipeline import StageReport, TensorInfo, recorder, registry, stage


HOST = "127.0.0.1"
PORT = 5000

# 브라우저로 집계를 보내는 간격(초)
PUSH_INTERVAL = 0.2

# 한 번 보낼 때 담을 최대 로그 줄 수
MAX_LOG_LINES = 80

# 저장 이력으로 보여 줄 최근 장수
HISTORY_LIMIT = 10

logger = logging.getLogger("step8")


# ============================================================
# 1. 명령줄 옵션
# ============================================================

def parse_args(argv=None):
    """
    <summary>명령줄 옵션을 읽는다. STEP 7과 같은 옵션에서 --mode만 뺐다.</summary>
    <param name="argv">옵션 문자열 목록. None이면 실제 명령줄을 쓴다.</param>
    <input>명령줄 문자열.</input>
    <output>argparse.Namespace 객체.</output>
    <reason>옵션 이름을 STEP 7과 똑같이 맞춰야, STEP 7에서 기록한 FPS 표와 같은 조건으로 비교할 수 있다.</reason>
    """
    parser = argparse.ArgumentParser(
        description="STEP 8: 파이프라인 대시보드",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0", "mps"], default=config.DEVICE, help="추론 장치")
    parser.add_argument("--imgsz", type=int, default=None, help="모델 입력 크기 (생략하면 CPU 320, GPU 640)")
    parser.add_argument("--conf", type=float, default=step7.DEFAULT_CONF, help="신뢰도 기준 (0~1)")
    parser.add_argument("--skip", type=int, default=None, help="N프레임마다 한 번 추론 (생략하면 CPU 2, GPU 1)")
    parser.add_argument("--model", default=config.MODEL_PATH, help="모델 파일 (.pt)")
    parser.add_argument("--openvino", action="store_true", help="<모델이름>_openvino_model 폴더가 있으면 OpenVINO로 실행")
    parser.add_argument("--classes", nargs="+", default=None, metavar="NAME", help="이 클래스만 탐지. 예: --classes person cup")

    args = parser.parse_args(argv)

    if args.imgsz is not None and args.imgsz < 32:
        parser.error("--imgsz는 32 이상이어야 합니다. (보통 320, 416, 640)")

    if args.skip is not None and args.skip < 1:
        parser.error("--skip은 1 이상이어야 합니다.")

    if not 0.0 < args.conf < 1.0:
        parser.error("--conf는 0과 1 사이여야 합니다. 예: 0.4")

    return args


# ============================================================
# 2. ultralytics 내부 엿보기
# ============================================================

@dataclass
class YoloCall:
    """
    <summary>model.predict() 한 번의 결과와, 그 안에서 엿본 텐서 모양을 담는다.</summary>
    <param name="result">Results 객체(results[0]).</param>
    <param name="speed">results[0].speed. preprocess, inference, postprocess 시간(ms).</param>
    <param name="call_start">predict()를 부른 시각(perf_counter 초).</param>
    <param name="tensor_info">전처리가 만든 입력 텐서의 모양.</param>
    <param name="raw_info">신경망이 낸 날것 출력의 모양.</param>
    <input>run_yolo()가 만든다.</input>
    <output>preprocess, inference, postprocess 단계 함수가 나눠 읽는다.</output>
    <reason>한 번의 호출 결과를 세 단계가 나눠 써야 하므로 한 객체에 모아 넘긴다.</reason>
    """

    result: object
    speed: dict
    call_start: float
    tensor_info: object
    raw_info: object


class PredictorProbe:
    """
    <summary>ultralytics 예측기의 preprocess, inference, postprocess 메서드를 감싸, 지금 어느 단계인지와 텐서 모양을 엿본다.</summary>
    <input>ensure(model)로 감싸기를 설치한다.</input>
    <output>tensor_info, raw_info에 마지막 호출의 모양이 남는다.</output>
    <reason>
    시간은 results[0].speed를 쓰지만, "지금 이 순간 활성 단계"와 "텐서 모양"은 speed에 없다.
    예측기 객체의 메서드를 얇게 감싸면 ultralytics 코드를 고치지 않고도 내부 흐름을 볼 수 있다.
    예측기는 설정이 바뀌면 새로 만들어지므로, 부를 때마다 감싸기가 남아 있는지 확인한다.
    </reason>
    """

    MARK = "_step8_probe"

    def __init__(self):
        """
        <summary>엿본 값을 담을 자리를 만든다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>아직 한 번도 호출되지 않았으면 None으로 남는다.</reason>
        """
        self.tensor_info = None
        self.raw_info = None

    def ensure(self, model):
        """
        <summary>예측기에 감싸기가 없으면 설치한다.</summary>
        <param name="model">YOLO 모델 객체.</param>
        <input>모델.</input>
        <output>없음.</output>
        <reason>예측기는 첫 predict() 때 만들어지므로, 그 전에는 설치할 대상이 없다.</reason>
        """
        predictor = getattr(model, "predictor", None)

        if predictor is None or getattr(predictor, self.MARK, False):
            return

        original_pre = predictor.preprocess
        original_inf = predictor.inference
        original_post = predictor.postprocess

        def preprocess(im):
            """
            <summary>원래 전처리를 부르기 전에 활성 단계를 바꾸고, 결과 텐서의 모양을 남긴다.</summary>
            <param name="im">ultralytics가 넘기는 원본 이미지 목록.</param>
            <input>이미지 목록.</input>
            <output>원래 전처리의 결과 텐서 그대로.</output>
            <reason>결과를 바꾸지 않고 엿보기만 해야 탐지 결과가 STEP 7과 똑같이 유지된다.</reason>
            """
            recorder.set_active("preprocess")
            out = original_pre(im)
            self.tensor_info = TensorInfo.of(out)
            return out

        def inference(im, *args, **kwargs):
            """
            <summary>원래 신경망 계산을 부르기 전에 활성 단계를 바꾸고, 날것 출력의 모양을 남긴다.</summary>
            <param name="im">전처리가 만든 입력 텐서.</param>
            <input>입력 텐서.</input>
            <output>원래 신경망 출력 그대로.</output>
            <reason>출력 모양을 보면 모델이 한 번에 몇 개의 후보 상자를 내는지 알 수 있다.</reason>
            """
            recorder.set_active("inference")
            out = original_inf(im, *args, **kwargs)
            self.raw_info = TensorInfo.of(out)
            return out

        def postprocess(preds, img, orig_imgs, *args, **kwargs):
            """
            <summary>원래 후처리를 부르기 전에 활성 단계를 바꾼다.</summary>
            <param name="preds">신경망 날것 출력.</param>
            <param name="img">입력 텐서.</param>
            <param name="orig_imgs">원본 이미지 목록.</param>
            <input>후처리 인자 그대로.</input>
            <output>원래 후처리의 결과(Results 목록) 그대로.</output>
            <reason>활성 단계 표시만 바꾸고 계산에는 손대지 않는다.</reason>
            """
            recorder.set_active("postprocess")
            return original_post(preds, img, orig_imgs, *args, **kwargs)

        predictor.preprocess = preprocess
        predictor.inference = inference
        predictor.postprocess = postprocess
        setattr(predictor, self.MARK, True)


def run_yolo(detector, probe, frame):
    """
    <summary>model.predict()를 한 번 부르고, 세 단계 이벤트를 만들 재료(YoloCall)를 돌려준다.</summary>
    <param name="detector">준비된 STEP 7 Detector 객체.</param>
    <param name="probe">PredictorProbe 객체.</param>
    <param name="frame">BGR 프레임.</param>
    <input>탐지기, 엿보기 도구, 프레임.</input>
    <output>YoloCall 객체.</output>
    <reason>이 함수는 단계가 아니다. 한 번의 호출 안에 세 단계가 숨어 있어서, 호출은 여기서 하고 기록은 세 단계 함수가 나눠 한다.</reason>
    """
    settings = detector.settings
    probe.ensure(detector.model)

    call_start = time.perf_counter()

    try:
        results = detector.model.predict(
            source=frame,
            imgsz=settings.imgsz,
            conf=settings.conf,
            device=settings.device,
            classes=detector.class_ids,
            verbose=False,
            **step7.precision_options(settings.half),
        )
    finally:
        recorder.set_active(None)

    probe.ensure(detector.model)
    result = results[0]

    return YoloCall(
        result=result,
        speed=result.speed,
        call_start=call_start,
        tensor_info=probe.tensor_info,
        raw_info=probe.raw_info,
    )


# ============================================================
# 3. 파이프라인 단계 (docstring이 그대로 플로우차트 설명이 된다)
# ============================================================

@stage("capture")
def capture(camera):
    """
    <summary>카메라 스레드가 읽어 둔 최신 프레임 한 장을 가져온다.</summary>
    <param name="camera">실행 중인 Camera 객체. 웹캠은 서버가 켜질 때 한 번만 열린다.</param>
    <input>Camera 객체.</input>
    <output>BGR 프레임. 모양 (480, 640, 3), 자료형 uint8, 값 0~255.</output>
    <reason>
    카메라 읽기(cap.read)는 다음 프레임이 올 때까지 기다리는 느린 일이라 별도 스레드가 맡는다(STEP 2).
    이 단계는 그 스레드가 준비해 둔 프레임을 복사만 하므로 거의 0 ms다.
    처리가 카메라보다 느리면 그사이 들어온 프레임은 버려지고, 로그에 프레임 드롭 경고가 남는다.
    </reason>
    """
    _, frame = camera.get_frame()

    return frame


@stage("preprocess", external=True)
def preprocess(frame, call):
    """
    <summary>YOLO 내부 전처리: 프레임을 모델이 먹을 수 있는 텐서로 바꾼다.</summary>
    <param name="frame">BGR 프레임.</param>
    <param name="call">model.predict() 한 번의 결과(YoloCall).</param>
    <input>BGR 프레임 (480, 640, 3) uint8.</input>
    <output>입력 텐서 (1, 3, 세로, 가로) float32. 예: imgsz 320이면 (1, 3, 256, 320).</output>
    <reason>
    비율을 지키며 imgsz에 맞게 줄이고(letterbox), BGR을 RGB로, (세로, 가로, 채널)을 (채널, 세로, 가로)로 바꾸고,
    0~255 정수를 0~1 실수로 나눈다. 맨 앞의 1은 "사진 1장 묶음(batch)"이라는 뜻이다.
    이 일은 model.predict() 안에서 일어나므로, 시간은 results[0].speed의 preprocess 값을 그대로 쓴다.
    </reason>
    """
    return StageReport(output=call.tensor_info, ms=call.speed["preprocess"], start=call.call_start)


@stage("inference", external=True)
def inference(tensor_info, call):
    """
    <summary>YOLO 신경망 계산: 입력 텐서를 받아 후보 상자들의 점수표를 만든다.</summary>
    <param name="tensor_info">전처리가 만든 입력 텐서의 모양.</param>
    <param name="call">model.predict() 한 번의 결과(YoloCall).</param>
    <input>입력 텐서 (1, 3, 세로, 가로) float32.</input>
    <output>날것 출력 텐서 (1, 84, 후보 수). 84 = 상자 좌표 4개 + 클래스 80개의 점수. 후보 수는 입력 크기에 따라 달라서 (1, 3, 256, 320)이면 1680개다.</output>
    <reason>
    후보 수 1680은 입력을 8배, 16배, 32배로 줄인 세 격자의 칸 수를 더한 값이다(32x40 + 16x20 + 8x10).
    격자 칸마다 "여기에 물체가 있다면 이런 상자, 이런 클래스"라는 후보를 하나씩 낸다.
    수백만 번의 곱셈과 덧셈이 일어나는 곳이라, CPU에서는 보통 전체 시간의 대부분을 차지한다.
    GPU로 바꾸면 이 단계가 가장 크게 줄어든다.
    model.predict() 안에서 일어나므로 시간은 results[0].speed의 inference 값을 쓴다.
    </reason>
    """
    start = call.call_start + call.speed["preprocess"] / 1000

    return StageReport(output=call.raw_info, ms=call.speed["inference"], start=start)


@stage("postprocess", external=True)
def postprocess(raw_info, call):
    """
    <summary>YOLO 내부 후처리: 수천 개의 후보 중 신뢰도 기준을 넘고 서로 겹치지 않는 상자만 남기고, 좌표를 원본 프레임 크기로 되돌린다.</summary>
    <param name="raw_info">신경망 날것 출력의 모양.</param>
    <param name="call">model.predict() 한 번의 결과(YoloCall).</param>
    <input>날것 출력 텐서 (1, 84, 후보 수).</input>
    <output>탐지 결과. 상자 좌표 (N, 4), 신뢰도 (N,), 클래스 번호 (N,). 로그에는 "N boxes"로 보인다.</output>
    <reason>
    conf보다 낮은 후보를 버리고, 같은 물체를 가리키는 겹친 상자는 가장 확실한 하나만 남긴다(NMS, 비최대 억제).
    그다음 letterbox로 줄였던 좌표를 다시 (480, 640) 기준으로 늘린다.
    로그의 in=(1,84,1680)이 out=몇 boxes로 줄어드는 것이 바로 이 단계다.
    model.predict() 안에서 일어나므로 시간은 results[0].speed의 postprocess 값을 쓴다.
    GPU 텐서를 numpy로 꺼내 두는 일까지 여기서 끝내, 다음 단계는 GPU를 몰라도 되게 한다.
    </reason>
    """
    boxes = call.result.boxes

    detections = step7.Detections(
        boxes=boxes.xyxy.cpu().numpy().astype(int),
        confs=boxes.conf.cpu().numpy(),
        classes=boxes.cls.cpu().numpy().astype(int),
    )
    start = call.call_start + (call.speed["preprocess"] + call.speed["inference"]) / 1000

    return StageReport(output=detections, ms=call.speed["postprocess"], start=start)


@stage("visualize")
def visualize(frame, detections, names, frame_no):
    """
    <summary>프레임 위에 탐지 상자, 라벨, 프레임 번호를 그린다.</summary>
    <param name="frame">BGR 프레임. 이 배열 위에 바로 그린다.</param>
    <param name="detections">탐지 결과(Detections).</param>
    <param name="names">클래스 번호 → 이름 딕셔너리.</param>
    <param name="frame_no">이 프레임의 파이프라인 번호.</param>
    <input>BGR 프레임과 탐지 결과.</input>
    <output>상자가 그려진 BGR 프레임. 모양은 입력과 같다.</output>
    <reason>
    STEP 5·6에서 익힌 cv2.rectangle과 cv2.putText를 그대로 쓴다.
    왼쪽 아래에 #프레임 번호를 찍어 두면, 화면의 영상과 로그 콘솔의 #frame 줄을 서로 맞춰 볼 수 있다.
    </reason>
    """
    step7.draw_detections(frame, detections, names)

    label = f"#{frame_no:04d}"
    height = frame.shape[0]
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, height - text_height - 14), (text_width + 16, height), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

    return frame


@stage("output")
def output(frame, board):
    """
    <summary>그림이 그려진 프레임을 JPEG로 압축해, 브라우저 스트림과 촬영 버튼이 가져갈 자리에 올려 둔다.</summary>
    <param name="frame">상자가 그려진 BGR 프레임.</param>
    <param name="board">결과를 올려 둘 ResultBoard 객체.</param>
    <input>BGR 프레임 (480, 640, 3) uint8.</input>
    <output>JPEG 바이트. 크기는 보통 30~60KB.</output>
    <reason>
    JPEG 압축은 한 번만 하고, 브라우저 접속이 여럿이어도 같은 바이트를 나눠 보낸다(STEP 7).
    약 900KB인 원본 배열이 수십 KB로 줄어들어 네트워크로 보내기 쉬워진다.
    </reason>
    """
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])

    if not ok:
        raise RuntimeError("JPEG 변환에 실패했습니다")

    jpeg = buffer.tobytes()
    board.publish(frame, jpeg)

    return jpeg


# ============================================================
# 4. 파이프라인 스레드
# ============================================================

class ResultBoard:
    """
    <summary>output 단계가 올려 둔 최신 결과(프레임, JPEG)를 웹 요청들이 가져가는 게시판.</summary>
    <input>publish()로 올리고 read_jpeg(), read_frame()으로 가져간다.</input>
    <output>최신 결과 번호와 JPEG, 또는 프레임 복사본.</output>
    <reason>파이프라인 스레드가 올리는 동안 웹 스레드가 읽으므로 lock으로 보호한다.</reason>
    """

    def __init__(self):
        """
        <summary>빈 게시판을 만든다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>result_id는 새 결과가 올라올 때마다 1씩 커져, 스트림이 같은 그림을 두 번 보내지 않게 한다.</reason>
        """
        self.lock = threading.Lock()
        self.frame = None
        self.jpeg = None
        self.result_id = 0

    def publish(self, frame, jpeg):
        """
        <summary>새 결과를 올린다.</summary>
        <param name="frame">상자가 그려진 BGR 프레임.</param>
        <param name="jpeg">그 프레임의 JPEG 바이트.</param>
        <input>프레임과 JPEG.</input>
        <output>없음.</output>
        <reason>프레임과 JPEG를 함께 바꿔야 촬영 버튼이 스트림과 같은 순간을 저장한다.</reason>
        """
        with self.lock:
            self.frame = frame
            self.jpeg = jpeg
            self.result_id += 1

    def read_jpeg(self):
        """
        <summary>최신 결과 번호와 JPEG를 돌려준다.</summary>
        <input>없음.</input>
        <output>(결과 번호, JPEG bytes) 튜플.</output>
        <reason>bytes는 바뀌지 않는 값이라 복사하지 않아도 안전하다.</reason>
        """
        with self.lock:
            return self.result_id, self.jpeg

    def read_frame(self):
        """
        <summary>최신 결과 프레임의 복사본을 돌려준다.</summary>
        <input>없음.</input>
        <output>BGR 이미지 또는 None.</output>
        <reason>촬영 저장은 JPEG를 다시 풀지 않고 원래 배열을 고화질로 저장한다.</reason>
        """
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()


class PipelineWorker:
    """
    <summary>별도 스레드에서 새 카메라 프레임마다 여섯 단계를 차례로 실행한다.</summary>
    <param name="camera">실행 중인 Camera 객체.</param>
    <param name="detector">준비된 Detector 객체.</param>
    <param name="board">ResultBoard 객체.</param>
    <input>start()로 시작한다.</input>
    <output>단계를 지날 때마다 pipeline.recorder에 이벤트가 쌓이고, board에 결과가 올라간다.</output>
    <reason>
    STEP 7의 FrameProcessor와 같은 일을 하지만, 각 단계를 @stage 함수로 나눠 불러 단계별 기록이 남게 했다.
    skip 규칙으로 추론을 건너뛴 프레임은 직전 탐지 결과를 재사용하고 WARN을 남긴다.
    </reason>
    """

    def __init__(self, camera, detector, board):
        """
        <summary>스레드가 쓸 값을 준비한다.</summary>
        <param name="camera">Camera 객체.</param>
        <param name="detector">Detector 객체.</param>
        <param name="board">ResultBoard 객체.</param>
        <input>카메라, 탐지기, 게시판.</input>
        <output>없음.</output>
        <reason>error 값을 두어 스레드 안의 문제를 웹 페이지에도 보여 준다.</reason>
        """
        self.camera = camera
        self.detector = detector
        self.board = board
        self.probe = PredictorProbe()
        self.running = False
        self.thread = None
        self.error = ""
        self.frame_no = 0
        self.last_camera_id = 0
        self.last_detections = None
        self.paused = False
        self.frame_lock = threading.Lock()

    def start(self):
        """
        <summary>파이프라인 스레드를 시작한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>daemon 스레드로 만들면 메인 프로그램이 끝날 때 함께 정리된다.</reason>
        """
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        <summary>파이프라인 스레드를 멈춘다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>카메라를 해제하기 전에 먼저 멈춰야 해제된 카메라를 읽지 않는다.</reason>
        """
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

    def _loop(self):
        """
        <summary>새 카메라 프레임이 오면 process_frame()을 부르는 일을 반복한다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>
        새 프레임이 없을 때는 잠깐 쉬어 CPU를 아낀다. 이때 플로우차트에는 활성 단계가 없다(대기 중).
        카메라를 끈 동안(paused)은 처리를 멈추고 기다리기만 한다. 스레드를 없애지 않아야 다시 켤 때 바로 이어진다.
        프레임 한 장의 처리는 frame_lock 안에서 해서, 처리 도중에 카메라가 해제되지 않게 한다.
        </reason>
        """
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            if not self.camera.running:
                self._fail("카메라 연결이 끊겼습니다. 서버를 다시 시작하세요.")
                break

            camera_id = self.camera.frame_id

            if camera_id == self.last_camera_id:
                time.sleep(0.002)
                continue

            try:
                with self.frame_lock:
                    if not self.paused:
                        self.process_frame(camera_id)
            except Exception as error:
                self._fail(f"파이프라인 처리 중 문제가 생겼습니다: {error}")
                break

    def _fail(self, message):
        """
        <summary>오류를 콘솔, 로그 콘솔, 웹 페이지에 남기고 스레드를 멈춘다.</summary>
        <param name="message">오류 문장.</param>
        <input>문자열.</input>
        <output>없음.</output>
        <reason>스레드 안의 오류는 아무도 보지 못한 채 사라지기 쉬우므로 세 곳에 모두 남긴다.</reason>
        """
        print(f"[오류] {message}")
        recorder.warn(message)
        self.error = message
        self.running = False

    def process_frame(self, camera_id):
        """
        <summary>프레임 한 장을 capture → preprocess → inference → postprocess → visualize → output 순서로 처리한다.</summary>
        <param name="camera_id">카메라 스레드가 매긴 이 프레임의 번호.</param>
        <input>카메라 프레임 번호.</input>
        <output>없음. 이벤트와 결과가 기록된다.</output>
        <reason>
        카메라 번호가 2 이상 건너뛰었다면, 처리하는 동안 카메라가 보낸 프레임을 놓친 것이다(프레임 드롭).
        skip=N이면 N장에 한 장만 추론하고, 나머지는 직전 상자를 다시 그린다.
        </reason>
        """
        settings = self.detector.settings
        self.frame_no += 1
        recorder.begin_frame(self.frame_no)

        if self.last_camera_id > 0:
            dropped = camera_id - self.last_camera_id - 1

            if dropped > 0:
                recorder.warn(f"프레임 드롭: 처리하는 동안 카메라 프레임 {dropped}장을 건너뜀", "capture")

        self.last_camera_id = camera_id

        frame = capture(self.camera)

        should_infer = self.last_detections is None or (self.frame_no - 1) % settings.skip == 0

        if should_infer:
            recorder.mark_inference(False)
            call = run_yolo(self.detector, self.probe, frame)
            tensor_info = preprocess(frame, call)
            raw_info = inference(tensor_info, call)
            detections = postprocess(raw_info, call)
            self.last_detections = detections
        else:
            recorder.mark_inference(True)
            detections = self.last_detections
            recorder.warn(f"skip={settings.skip}: 추론 생략, 직전 결과 {len(detections)}개 재사용", "inference")

        annotated = visualize(frame, detections, self.detector.names, self.frame_no)
        output(annotated, self.board)


def make_off_frame(width, height):
    """
    <summary>카메라를 껐을 때 스트림에 보여 줄 안내 그림을 만든다.</summary>
    <param name="width">가로 크기.</param>
    <param name="height">세로 크기.</param>
    <input>크기.</input>
    <output>(BGR 그림, JPEG 바이트) 튜플.</output>
    <reason>마지막 프레임이 멈춘 채 남아 있으면 "꺼진 것"인지 "멈춘 것"인지 헷갈리므로, 꺼짐을 그림으로 분명히 보여 준다.</reason>
    """
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    text = "Camera OFF"
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, text, ((width - text_width) // 2, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode(".jpg", frame)

    return frame, buffer.tobytes()


class CameraSwitch:
    """
    <summary>서버를 끄지 않고 웹캠만 켜고 끈다. 끄면 카메라 장치를 완전히 해제해 웹캠 표시등도 꺼진다.</summary>
    <param name="camera">STEP 7 Camera 객체.</param>
    <param name="worker">PipelineWorker 객체.</param>
    <param name="board">ResultBoard 객체.</param>
    <input>turn_on(), turn_off()로 조작한다.</input>
    <output>is_on 값이 지금 상태다.</output>
    <reason>
    수업 중 잠깐 카메라를 끄려고 서버를 내리면, 모델 불러오기와 워밍업을 처음부터 다시 해야 한다.
    카메라만 해제하고 모델과 파이프라인 스레드는 그대로 두면, 다시 켤 때 1~2초면 이어진다.
    끄는 중에 처리 중이던 프레임이 끝나기를 기다린 뒤(frame_lock) 해제해야 해제된 카메라를 읽는 오류가 없다.
    </reason>
    """

    def __init__(self, camera, worker, board):
        """
        <summary>스위치를 준비한다. 처음 상태는 켜짐이다.</summary>
        <param name="camera">Camera 객체.</param>
        <param name="worker">PipelineWorker 객체.</param>
        <param name="board">ResultBoard 객체.</param>
        <input>카메라, 작업자, 게시판.</input>
        <output>없음.</output>
        <reason>켜기와 끄기 요청이 동시에 와도 하나씩 처리되도록 lock을 둔다.</reason>
        """
        self.camera = camera
        self.worker = worker
        self.board = board
        self.lock = threading.Lock()
        self.is_on = True

    def turn_off(self):
        """
        <summary>파이프라인을 멈추고 카메라를 해제한다.</summary>
        <input>없음.</input>
        <output>(성공 여부, 안내 문장) 튜플.</output>
        <reason>이미 꺼져 있으면 아무 일도 하지 않는다. 같은 링크를 두 번 눌러도 안전하다.</reason>
        """
        with self.lock:
            if not self.is_on:
                return True, "이미 꺼져 있습니다."

            self.worker.paused = True

            with self.worker.frame_lock:
                self.camera.release()

            with self.camera.lock:
                self.camera.frame = None

            frame, jpeg = make_off_frame(config.FRAME_WIDTH, config.FRAME_HEIGHT)
            self.board.publish(frame, jpeg)
            self.is_on = False

        recorder.mark_inference(False)
        recorder.note("카메라 꺼짐 (장치 해제)", "capture")
        print("[정보] 웹캠을 껐습니다.")

        return True, "카메라를 껐습니다."

    def turn_on(self):
        """
        <summary>카메라를 다시 열고 파이프라인을 이어서 돌린다.</summary>
        <input>없음.</input>
        <output>(성공 여부, 안내 문장) 튜플.</output>
        <reason>
        다른 프로그램이 그사이 카메라를 가져갔을 수 있으므로, 열기에 실패하면 꺼진 상태로 두고 이유를 알려 준다.
        last_camera_id를 0으로 되돌려, 다시 켠 첫 프레임을 프레임 드롭으로 잘못 세지 않게 한다.
        </reason>
        """
        with self.lock:
            if self.is_on:
                return True, "이미 켜져 있습니다."

            if not self.worker.running:
                return False, "파이프라인이 멈춰 있어 켤 수 없습니다. 서버를 다시 시작하세요."

            if not self.camera.start():
                return False, "카메라를 열 수 없습니다. 다른 프로그램이 쓰고 있지 않은지 확인하세요."

            if not self.camera.wait_first_frame(timeout=5):
                self.camera.release()
                return False, "카메라는 열렸지만 영상이 들어오지 않습니다."

            self.worker.last_camera_id = 0
            self.worker.paused = False
            self.is_on = True

        recorder.note("카메라 켜짐", "capture")
        print("[정보] 웹캠을 켰습니다.")

        return True, "카메라를 켰습니다."


# ============================================================
# 5. 웹 서버
# ============================================================

def generate_detect_mjpeg(board, worker):
    """
    <summary>게시판의 최신 JPEG를 MJPEG 스트림으로 내보낸다.</summary>
    <param name="board">ResultBoard 객체.</param>
    <param name="worker">PipelineWorker 객체. 멈추면 스트림도 끝난다.</param>
    <input>게시판과 작업자.</input>
    <output>MJPEG 조각(bytes) 제너레이터.</output>
    <reason>새 결과가 올라왔을 때만 보내, 같은 그림을 반복해 보내지 않는다.</reason>
    """
    last_id = 0

    while worker.running:
        result_id, jpeg = board.read_jpeg()

        if jpeg is None or result_id == last_id:
            time.sleep(0.01)
            continue

        last_id = result_id

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def generate_events(worker, settings, switch):
    """
    <summary>200ms마다 링버퍼를 집계해 Server-Sent Events 형식으로 브라우저에 보낸다.</summary>
    <param name="worker">PipelineWorker 객체.</param>
    <param name="settings">RunSettings 객체.</param>
    <param name="switch">CameraSwitch 객체. 카메라 켜짐 여부를 함께 보낸다.</param>
    <input>작업자, 설정, 카메라 스위치.</input>
    <output>"data: (JSON 한 줄)" 뒤에 빈 줄이 붙은 문자열을 하나씩 내보내는 제너레이터.</output>
    <reason>
    WebSocket 대신 SSE를 쓰는 이유는 세 가지다.
    첫째, 이 대시보드는 서버 → 브라우저 한 방향으로만 보내면 된다.
    둘째, SSE는 평범한 HTTP 응답이라 Flask가 기본으로 지원한다(제너레이터를 돌려주기만 하면 된다).
    셋째, flask-sock 같은 추가 라이브러리가 필요 없고, 브라우저의 EventSource는 연결이 끊기면 알아서 다시 붙는다.
    매 프레임 이벤트를 그대로 보내면 1초에 수백 개가 되어 CPU 노트북에 부담이 크므로, 200ms마다 모아서 보낸다.
    </reason>
    """
    since_seq = max(0, recorder.latest_seq() - MAX_LOG_LINES)
    static = {
        "device": settings.device_label(),
        "imgsz": settings.imgsz,
        "conf": settings.conf,
        "skip": settings.skip,
    }

    yield "retry: 2000\n\n"

    while True:
        snapshot = recorder.snapshot(since_seq, MAX_LOG_LINES)
        since_seq = snapshot["last_seq"]
        snapshot.update(static)
        snapshot["error"] = worker.error
        snapshot["camera_on"] = switch.is_on

        yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"

        time.sleep(PUSH_INTERVAL)


def create_app(camera, worker, board, store, settings, switch):
    """
    <summary>대시보드 페이지, 탐지 스트림, SSE, 단계 메타데이터, 촬영, 카메라 켜기/끄기 주소를 가진 Flask 앱을 만든다.</summary>
    <param name="camera">Camera 객체.</param>
    <param name="worker">PipelineWorker 객체.</param>
    <param name="board">ResultBoard 객체.</param>
    <param name="store">STEP 7의 CaptureStore 객체.</param>
    <param name="settings">RunSettings 객체.</param>
    <param name="switch">CameraSwitch 객체.</param>
    <input>카메라, 작업자, 게시판, 저장소, 설정, 카메라 스위치.</input>
    <output>Flask 앱 객체.</output>
    <reason>STEP 3·7의 주소(/detect_feed, /video_feed, /capture, /captures)는 그대로 두고 /events, /api/pipeline, /camera를 더한다.</reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>templates/dashboard.html을 그려서 돌려준다.</summary>
        <input>GET / 요청.</input>
        <output>HTML 응답.</output>
        <reason>화면 구성(HTML), 모양(CSS), 동작(JS)을 파일로 나눠, 각각 따로 고칠 수 있게 한다.</reason>
        """
        return render_template("dashboard.html", history_limit=HISTORY_LIMIT)

    @app.route("/api/pipeline")
    def api_pipeline():
        """
        <summary>단계 목록과 docstring에서 읽은 설명을 JSON으로 돌려준다.</summary>
        <input>GET /api/pipeline 요청.</input>
        <output>[{id, label, summary, input, output, reason, params, function, external, parsed}] JSON.</output>
        <reason>페이지는 이 목록만 보고 플로우차트를 그린다. 단계 함수의 docstring을 고치면 페이지 설명도 바로 바뀐다.</reason>
        """
        return jsonify(registry.metadata())

    @app.route("/events")
    def events():
        """
        <summary>집계 데이터를 Server-Sent Events로 계속 보낸다.</summary>
        <input>GET /events 요청 (브라우저의 EventSource).</input>
        <output>text/event-stream 스트리밍 응답.</output>
        <reason>Cache-Control: no-cache를 붙여야 중간에서 응답을 모아 두지 않고 바로바로 흘려보낸다.</reason>
        """
        return Response(
            generate_events(worker, settings, switch),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/camera")
    def camera_page():
        """
        <summary>웹캠 켜기/끄기 버튼만 있는 간단한 페이지를 돌려준다.</summary>
        <input>GET /camera 요청.</input>
        <output>HTML 응답.</output>
        <reason>대시보드를 열지 않고도 링크 하나로 카메라를 켜고 끌 수 있게 한다. 휴대폰 즐겨찾기에 넣어 두기도 쉽다.</reason>
        """
        message = request.args.get("msg", "")

        return render_template("camera.html", is_on=switch.is_on, message=message)

    @app.route("/camera/status")
    def camera_status():
        """
        <summary>카메라가 켜져 있는지 JSON으로 돌려준다.</summary>
        <input>GET /camera/status 요청.</input>
        <output>{"on": true 또는 false}</output>
        <reason>다른 프로그램이나 스크립트가 상태만 확인할 때 쓴다.</reason>
        """
        return jsonify(on=switch.is_on)

    @app.route("/camera/<action>", methods=["POST"])
    def camera_switch(action):
        """
        <summary>카메라를 켜거나 끈다.</summary>
        <param name="action">"on" 또는 "off".</param>
        <input>POST /camera/on 또는 POST /camera/off 요청.</input>
        <output>fetch로 부르면 {"ok", "on", "message"} JSON, 페이지 버튼으로 부르면 /camera로 되돌아간다.</output>
        <reason>
        카메라 상태를 바꾸는 동작이라 GET이 아닌 POST로 받는다(STEP 3).
        GET 링크로 만들면 브라우저가 주소를 미리 불러오기만 해도 카메라가 꺼질 수 있다.
        </reason>
        """
        if action == "on":
            ok, message = switch.turn_on()
        elif action == "off":
            ok, message = switch.turn_off()
        else:
            abort(404)

        if request.accept_mimetypes.best == "application/json":
            return jsonify(ok=ok, on=switch.is_on, message=message), (200 if ok else 503)

        return redirect(url_for("camera_page", msg=message))

    @app.route("/detect_feed")
    def detect_feed():
        """
        <summary>탐지 결과 MJPEG 스트림.</summary>
        <input>GET /detect_feed 요청.</input>
        <output>multipart/x-mixed-replace 스트리밍 응답.</output>
        <reason>STEP 7과 같은 주소라 img 태그를 그대로 쓸 수 있다.</reason>
        """
        return Response(generate_detect_mjpeg(board, worker), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>원본 카메라 MJPEG 스트림.</summary>
        <input>GET /video_feed 요청.</input>
        <output>multipart/x-mixed-replace 스트리밍 응답.</output>
        <reason>탐지 결과와 나란히 열어 파이프라인이 원본보다 얼마나 늦는지 비교할 수 있게 남긴다.</reason>
        """
        return Response(step7.generate_raw_mjpeg(camera), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/capture", methods=["POST"])
    def capture_route():
        """
        <summary>현재 탐지 결과 프레임을 저장하고 파일 이름을 JSON으로 돌려준다.</summary>
        <input>POST /capture 요청.</input>
        <output>성공: {"filename", "url", "frame"} (200). 실패: {"error"} (503 또는 500).</output>
        <reason>저장한 순간의 프레임 번호를 함께 돌려주면, 로그 콘솔에서 그 프레임의 단계 기록을 찾아볼 수 있다.</reason>
        """
        if not switch.is_on:
            return jsonify(error="카메라가 꺼져 있습니다. 카메라를 켠 뒤 촬영하세요."), 503

        frame = board.read_frame()

        if frame is None or not worker.running:
            return jsonify(error="탐지 결과가 아직 없어 저장할 수 없습니다."), 503

        filename = store.save(frame)

        if filename is None:
            return jsonify(error="이미지를 저장하지 못했습니다."), 500

        return jsonify(filename=filename, url=f"/captures/{filename}", frame=worker.frame_no)

    @app.route("/captures")
    def capture_list():
        """
        <summary>최근 저장 이력을 JSON으로 돌려준다.</summary>
        <input>GET /captures 요청.</input>
        <output>{"captures": [{"filename", "url"}]} 최신순.</output>
        <reason>페이지를 새로 고쳐도 저장 이력이 남아 보이게 한다.</reason>
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
        if not step7.is_valid_capture_name(filename):
            abort(404)

        return send_from_directory(store.directory, filename, mimetype="image/jpeg")

    return app


# ============================================================
# 6. 시작
# ============================================================

def check_pipeline_docs():
    """
    <summary>서버를 켜기 전에 모든 단계의 docstring을 한 번 읽어, 파싱에 실패한 단계를 미리 알린다.</summary>
    <input>없음.</input>
    <output>파싱에 성공한 단계 수.</output>
    <reason>docstring에 "&amp;"나 짝이 맞지 않는 태그가 있으면 XML로 읽히지 않는다. 브라우저를 열기 전에 콘솔에서 알려 주면 고치기 쉽다.</reason>
    """
    items = registry.metadata()
    parsed = sum(1 for item in items if item["parsed"])
    print(f"[정보] 단계 설명(docstring) {parsed}/{len(items)}개를 읽었습니다.")

    return parsed


def run(argv=None):
    """
    <summary>옵션 해석 → 모델 준비 → 카메라와 파이프라인 시작 → Flask 서버 실행 순서로 동작한다.</summary>
    <param name="argv">명령줄 옵션 목록. None이면 실제 명령줄.</param>
    <input>명령줄 옵션.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 오류는 1.</output>
    <reason>
    모델 준비(다운로드, FP16 확인, 워밍업)를 카메라보다 먼저 끝낸다.
    종료는 시작의 반대 순서(파이프라인 → 카메라)로, atexit에 등록해 Ctrl+C로 끝내도 정리되게 한다.
    </reason>
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    args = parse_args(argv)
    settings = step7.resolve_settings(args)
    detector = step7.Detector(settings)

    if not detector.load():
        return 1

    frame_shape = (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)

    if not detector.prepare(frame_shape, args.classes):
        return 1

    step7.print_settings(settings, detector.class_ids, detector.names)
    check_pipeline_docs()

    try:
        store = step7.CaptureStore(step7.DETECTION_DIR)
    except OSError as error:
        print(f"[오류] 저장 폴더를 만들지 못했습니다: {error}")
        return 1

    camera = step7.Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if not camera.start():
        step7.print_camera_help(config.CAMERA_INDEX)
        return 1

    board = ResultBoard()
    worker = PipelineWorker(camera, detector, board)

    atexit.register(camera.release)
    atexit.register(worker.stop)

    if not camera.wait_first_frame(timeout=5):
        print("[오류] 카메라는 열렸지만 5초 동안 영상이 들어오지 않았습니다.")
        print("카메라 차단 스위치나 다른 프로그램의 사용 여부를 확인하세요.")
        return 1

    worker.start()
    switch = CameraSwitch(camera, worker, board)
    app = create_app(camera, worker, board, store, settings, switch)

    print(f"[정보] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print(f"[정보] 웹캠 켜기/끄기: http://{HOST}:{PORT}/camera")
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


if __name__ == "__main__":
    sys.exit(run())


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행 (pipeline.py, step7_realtime_yolo.py, templates\dashboard.html, static\ 폴더가 같은 곳에 있어야 한다)
#      python step8_pipeline_dashboard.py
#      python step8_pipeline_dashboard.py --device cuda:0
#      python step8_pipeline_dashboard.py --imgsz 640 --skip 1
# 3. 브라우저에서 http://127.0.0.1:5000 에 접속한다.
#    - 왼쪽: 탐지 결과 영상과 촬영 버튼 (영상 왼쪽 아래의 #번호가 로그의 #frame 번호다)
#    - 상단 바의 "카메라 끄기" 버튼으로 서버를 끄지 않고 웹캠만 끄고 켤 수 있다.
#      켜기/끄기만 있는 간단한 페이지: http://127.0.0.1:5000/camera
#    - 오른쪽 위: 플로우차트. 노드를 클릭하면 그 단계 함수의 docstring 설명이 보인다.
#    - 오른쪽 아래: 로그 콘솔. 일시정지와 INFO/DEBUG/WARN 필터가 있다.
# 4. 데이터만 보고 싶다면
#      http://127.0.0.1:5000/api/pipeline   단계 메타데이터 JSON
#      http://127.0.0.1:5000/events         200ms마다 오는 SSE 원본 (data: ... 줄)
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. CPU에서 inference는 전체 시간의 몇 %인가?
#    - 플로우차트 위의 "한 프레임 시간 구성" 막대와 각 노드 아래 ms · % 를 읽어 보자.
#    - 병목 표시가 붙은 노드는 어디인가? capture와 output은 몇 ms인가?
#    - --device cuda:0 으로 다시 실행하면 inference의 %는 어떻게 바뀌는가?
#      inference가 줄어들면 그다음 병목은 어느 노드가 되는가?
#
# 2. imgsz를 바꾸면 어느 노드의 숫자가 변하는가?
#    - --imgsz 320 과 --imgsz 640 으로 각각 실행해 노드별 ms를 표로 적자.
#        | 노드        | imgsz 320 | imgsz 640 |
#        |-------------|-----------|-----------|
#        | capture     |           |           |
#        | preprocess  |           |           |
#        | inference   |           |           |
#        | postprocess |           |           |
#        | visualize   |           |           |
#        | output      |           |           |
#    - DEBUG 로그에서 preprocess의 out= 모양은 어떻게 바뀌는가? (예: (1,3,256,320) → (1,3,480,640))
#    - imgsz와 상관없이 거의 그대로인 노드는 어디인가? 그 노드들은 무엇을 기준으로 일하는가?
#      (힌트: visualize와 output은 모델 입력이 아니라 카메라 원본 크기 480 x 640으로 일한다)
#
# 3. skip 값에 따라 로그의 WARN 빈도가 어떻게 바뀌는가?
#    - 로그 필터에서 WARN만 켜고 --skip 1, 2, 3 으로 실행해 보자.
#    - skip=2면 몇 프레임에 한 번 "추론 생략" WARN이 나오는가? skip=3이면?
#    - 추론을 건너뛴 프레임에서 플로우차트의 YOLO 내부 노드(preprocess, inference, postprocess)는 어떻게 보이는가?
#    - skip=1인데도 WARN이 나온다면 내용을 읽어 보자. "프레임 드롭"이라면 무슨 뜻일까?
#      (처리가 카메라보다 느려서, 처리하는 동안 들어온 프레임을 버렸다는 뜻이다)
#
# 4. 코드와 설명
#    - 플로우차트에서 inference 노드를 클릭해 상세 설명을 읽어 보자.
#      이 파일의 inference 함수 docstring과 글자가 똑같은가?
#    - docstring의 summary 한 줄을 고치고 서버를 다시 켜면 페이지 설명도 바뀌는가?
#    - docstring에 일부러 "<summary>" 닫는 태그를 지워 보자. 서버 콘솔에 어떤 경고가 나오고, 노드는 어떻게 보이는가?
