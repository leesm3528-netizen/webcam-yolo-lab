"""
<summary>파이프라인 단계 계측 도구. 단계 레지스트리, @stage 데코레이터, 이벤트 링버퍼, docstring 파싱을 담는다.</summary>
<reason>
"지금 프레임이 어느 단계를 지나고 있는가"를 보려면, 단계마다 시작과 끝을 기록해야 한다.
측정 코드를 단계 함수 안에 직접 넣으면 본래 처리 코드가 지저분해진다.
데코레이터 한 줄(@stage("inference"))로 감싸면 본래 코드는 그대로 두고 측정만 덧붙일 수 있다.
</reason>
"""

import functools
import inspect
import logging
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger("pipeline")


# 파이프라인 단계와 화면 이름. 이 순서가 곧 플로우차트의 위에서 아래 순서다.
STAGE_ORDER = [
    ("capture", "캡처"),
    ("preprocess", "전처리"),
    ("inference", "추론"),
    ("postprocess", "후처리"),
    ("visualize", "시각화"),
    ("output", "출력"),
]

# ultralytics의 model.predict() 한 번 안에서 일어나는 단계들
YOLO_INTERNAL_STAGES = ("preprocess", "inference", "postprocess")

# 이벤트 링버퍼 크기
EVENT_BUFFER_SIZE = 500

# 단계별 평균을 낼 최근 프레임 수
AVERAGE_FRAMES = 30

# FPS를 평균 낼 최근 프레임 수
FPS_WINDOW = 30

# docstring에서 읽을 태그들
DOC_TAGS = ("summary", "input", "output", "reason")


# ============================================================
# 1. 입력·출력 요약
# ============================================================

@dataclass
class TensorInfo:
    """
    <summary>텐서나 배열의 모양(shape)과 자료형(dtype)만 담은 가벼운 요약.</summary>
    <param name="shape">모양 튜플. 예: (1, 3, 256, 320)</param>
    <param name="dtype">자료형 이름. 예: "float32"</param>
    <input>TensorInfo.of(값)으로 만든다.</input>
    <output>summarize()가 "(1,3,256,320) float32" 같은 문자열로 바꾼다.</output>
    <reason>텐서 자체를 이벤트에 담으면 메모리를 크게 차지하고 GPU 메모리도 붙잡는다. 로그에는 모양과 자료형이면 충분하다.</reason>
    """

    shape: tuple
    dtype: str

    @classmethod
    def of(cls, value):
        """
        <summary>numpy 배열, torch 텐서, 또는 그것을 담은 리스트·튜플에서 첫 텐서의 모양을 꺼낸다.</summary>
        <param name="value">배열, 텐서, 리스트, 튜플 중 하나.</param>
        <input>요약할 값.</input>
        <output>TensorInfo. 모양을 알 수 없으면 None.</output>
        <reason>모델마다 출력이 텐서 하나일 수도, 텐서 묶음일 수도 있어서 첫 텐서를 찾아 내려간다.</reason>
        """
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            dtype = str(value.dtype).replace("torch.", "")
            return cls(tuple(int(n) for n in value.shape), dtype)

        if isinstance(value, (list, tuple)) and value:
            return cls.of(value[0])

        return None


@dataclass
class StageReport:
    """
    <summary>ultralytics 내부 단계처럼 직접 잴 수 없는 단계가 "걸린 시간과 결과"를 데코레이터에 알려 주는 보고서.</summary>
    <param name="output">이 단계의 출력(요약에 쓰인다).</param>
    <param name="ms">이 단계에 걸린 시간(밀리초).</param>
    <param name="start">이 단계가 시작된 시각(perf_counter 기준 초).</param>
    <input>external=True 단계 함수가 돌려준다.</input>
    <output>데코레이터가 이벤트로 기록한 뒤, output만 호출한 쪽에 돌려준다.</output>
    <reason>전처리·추론·후처리는 model.predict() 한 번 안에 숨어 있어, 함수 앞뒤 시각으로는 따로 잴 수 없다.</reason>
    """

    output: object
    ms: float
    start: float


def summarize(value):
    """
    <summary>단계의 입력이나 출력을 로그 한 칸에 들어갈 짧은 문자열로 바꾼다.</summary>
    <param name="value">요약할 값. 배열, 텐서 요약, 탐지 결과, 바이트 등.</param>
    <input>아무 값.</input>
    <output>예: "(480,640,3) uint8", "5 boxes", "jpeg 38KB"</output>
    <reason>로그의 in=, out= 칸을 보면 데이터가 단계를 지날 때 모양이 어떻게 바뀌는지 따라갈 수 있다.</reason>
    """
    if value is None:
        return "-"

    if isinstance(value, TensorInfo):
        return f"({','.join(str(n) for n in value.shape)}) {value.dtype}"

    if isinstance(value, (bytes, bytearray)):
        return f"jpeg {len(value) / 1024:.0f}KB"

    if hasattr(value, "classes") and hasattr(value, "__len__"):
        return f"{len(value)} boxes"

    info = TensorInfo.of(value)

    if info is not None:
        return summarize(info)

    if hasattr(value, "index") and hasattr(value, "get_frame"):
        return f"camera #{value.index}"

    return type(value).__name__


# ============================================================
# 2. 단계 레지스트리와 docstring → 메타데이터
# ============================================================

def _clean_text(element):
    """
    <summary>XML 요소 안의 글자를 모아 줄바꿈과 들여쓰기를 정리한다.</summary>
    <param name="element">ElementTree 요소.</param>
    <input>XML 요소.</input>
    <output>앞뒤 공백을 지운 문자열. 여러 줄은 한 줄씩 공백 하나로 이어 붙인다.</output>
    <reason>docstring은 사람이 읽기 좋게 줄을 나눠 쓰므로, 화면에 보여 줄 때는 문장으로 이어 준다.</reason>
    """
    text = "".join(element.itertext())
    lines = [line.strip() for line in text.strip().splitlines()]

    return " ".join(line for line in lines if line)


def parse_docstring(func):
    """
    <summary>함수의 XML docstring에서 summary, input, output, reason, param을 읽는다.</summary>
    <param name="func">docstring을 읽을 함수.</param>
    <input>함수 객체.</input>
    <output>(메타데이터 딕셔너리, 오류 메시지) 튜플. 성공하면 오류 메시지는 None.</output>
    <reason>
    docstring은 태그만 나열되어 있어 뿌리 요소가 없다. 앞뒤를 doc 태그로 감싸야 XML 한 덩어리로 읽힌다.
    inspect.getdoc은 들여쓰기를 정리해 주므로 클래스 안의 함수도 같은 방법으로 읽을 수 있다.
    </reason>
    """
    doc = inspect.getdoc(func)

    if not doc:
        return {}, "docstring이 없습니다"

    try:
        root = ET.fromstring(f"<doc>{doc}</doc>")
    except ET.ParseError as error:
        return {}, f"XML 파싱 실패: {error}"

    meta = {}

    for tag in DOC_TAGS:
        element = root.find(tag)
        meta[tag] = _clean_text(element) if element is not None else ""

    meta["params"] = [
        {"name": element.get("name", ""), "text": _clean_text(element)}
        for element in root.findall("param")
    ]

    return meta, None


class StageRegistry:
    """
    <summary>@stage로 등록된 단계 함수들을 모아 두고, 플로우차트용 메타데이터를 만든다.</summary>
    <input>stage 데코레이터가 register()를 부른다.</input>
    <output>metadata()가 /api/pipeline이 그대로 보낼 단계 목록을 돌려준다.</output>
    <reason>등록을 한곳에 모으면 웹 페이지가 코드의 단계 구성을 그대로 따라 그릴 수 있다. 단계를 하나 더해도 페이지 코드는 고칠 필요가 없다.</reason>
    """

    def __init__(self):
        """
        <summary>빈 레지스트리를 만든다.</summary>
        <input>없음.</input>
        <output>없음.</output>
        <reason>단계 이름 → (함수, 외부 측정 여부)를 딕셔너리로 들고 있는다.</reason>
        """
        self.stages = {}

    def register(self, stage_id, func, external):
        """
        <summary>단계 함수를 등록한다.</summary>
        <param name="stage_id">단계 이름. 예: "inference"</param>
        <param name="func">단계 함수(데코레이터로 감싸기 전 원래 함수).</param>
        <param name="external">ultralytics 내부에서 일어나 시간을 바깥에서 받아 오는 단계인지.</param>
        <input>단계 이름, 함수, 외부 측정 여부.</input>
        <output>없음.</output>
        <reason>STAGE_ORDER에 없는 이름은 오타일 가능성이 높아 경고를 남긴다.</reason>
        """
        known = {stage for stage, _ in STAGE_ORDER}

        if stage_id not in known:
            logger.warning("알 수 없는 단계 이름 '%s' (STAGE_ORDER에 없음)", stage_id)

        self.stages[stage_id] = (func, external)

    def metadata(self):
        """
        <summary>STAGE_ORDER 순서대로 단계 메타데이터 목록을 만든다.</summary>
        <input>없음.</input>
        <output>[{id, label, summary, input, output, reason, params, function, external, parsed}] 목록.</output>
        <reason>docstring을 읽지 못한 단계도 플로우차트에서 빠지지 않도록 label만이라도 채워 보내고, 서버 로그에 경고를 남긴다.</reason>
        """
        items = []

        for stage_id, label in STAGE_ORDER:
            item = {
                "id": stage_id,
                "label": label,
                "summary": "",
                "input": "",
                "output": "",
                "reason": "",
                "params": [],
                "function": None,
                "external": stage_id in YOLO_INTERNAL_STAGES,
                "parsed": False,
            }

            entry = self.stages.get(stage_id)

            if entry is None:
                logger.warning("단계 '%s'에 등록된 함수가 없어 이름만 표시합니다", stage_id)
                items.append(item)
                continue

            func, external = entry
            meta, error = parse_docstring(func)
            item["function"] = func.__name__
            item["external"] = external

            if error is None:
                item.update(meta)
                item["parsed"] = True
            else:
                logger.warning("단계 '%s'(%s)의 docstring을 읽지 못했습니다: %s", stage_id, func.__name__, error)

            items.append(item)

        return items


# ============================================================
# 3. 이벤트 기록 (링버퍼)
# ============================================================

class EventRecorder:
    """
    <summary>단계 통과 이벤트와 경고를 링버퍼(deque, 최대 500개)에 기록하고, 200ms마다 보낼 집계를 만든다.</summary>
    <param name="maxlen">링버퍼 크기.</param>
    <input>record_stage(), warn()으로 기록하고 snapshot()으로 집계한다.</input>
    <output>snapshot()이 단계별 평균 ms, 활성 단계, 새 로그, FPS를 담은 딕셔너리를 돌려준다.</output>
    <reason>
    deque(maxlen=500)은 가득 차면 가장 오래된 이벤트를 저절로 버려서, 오래 실행해도 메모리가 늘지 않는다.
    파이프라인 스레드가 쓰는 동안 웹 스레드가 읽으므로 lock으로 보호한다.
    lock 없이 읽으면 "deque mutated during iteration" 오류가 날 수 있다.
    </reason>
    """

    def __init__(self, maxlen=EVENT_BUFFER_SIZE):
        """
        <summary>빈 링버퍼와 상태 값을 준비한다.</summary>
        <param name="maxlen">링버퍼 크기.</param>
        <input>정수.</input>
        <output>없음.</output>
        <reason>seq는 이벤트마다 1씩 커지는 번호다. 브라우저가 "어디까지 받았는지"를 이 번호로 기억한다.</reason>
        """
        self.events = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.seq = 0
        self.frame = 0
        self.active = None
        self.inference_skipped = False
        self.output_times = deque(maxlen=FPS_WINDOW)

    def begin_frame(self, frame_no):
        """
        <summary>새 프레임 처리를 시작한다고 알린다.</summary>
        <param name="frame_no">파이프라인이 매기는 프레임 번호(1부터).</param>
        <input>프레임 번호.</input>
        <output>없음.</output>
        <reason>데코레이터는 어느 프레임의 이벤트인지 모르므로, 여기서 정한 번호를 이벤트마다 붙인다.</reason>
        """
        with self.lock:
            self.frame = frame_no

    def set_active(self, stage_id):
        """
        <summary>지금 실행 중인 단계를 바꾼다. None이면 쉬는 중(다음 카메라 프레임 대기)이다.</summary>
        <param name="stage_id">단계 이름 또는 None.</param>
        <input>단계 이름.</input>
        <output>없음.</output>
        <reason>200ms마다 집계할 때 이 값을 읽으면 "그 순간 어느 단계에 있었는지"가 플로우차트에 표시된다.</reason>
        """
        self.active = stage_id

    def mark_inference(self, skipped):
        """
        <summary>이번 프레임에서 추론을 건너뛰었는지 기록한다.</summary>
        <param name="skipped">True면 skip 규칙으로 추론을 생략했다.</param>
        <input>bool.</input>
        <output>없음.</output>
        <reason>플로우차트가 건너뛴 프레임에서 YOLO 내부 단계 노드를 흐리게 그릴 때 쓴다.</reason>
        """
        self.inference_skipped = skipped

    def _append(self, event):
        """
        <summary>이벤트에 번호와 시각을 붙여 링버퍼에 넣는다. lock을 잡은 상태에서만 부른다.</summary>
        <param name="event">이벤트 딕셔너리.</param>
        <input>딕셔너리.</input>
        <output>없음.</output>
        <reason>번호 붙이기와 넣기를 한 lock 안에서 해야 번호가 겹치거나 순서가 뒤바뀌지 않는다.</reason>
        """
        self.seq += 1
        event["seq"] = self.seq
        event["frame"] = self.frame
        event["wall"] = time.time()
        self.events.append(event)

    def record_stage(self, stage_id, start, end, ms, input_summary, output_summary):
        """
        <summary>단계 하나를 통과했다는 이벤트를 기록한다.</summary>
        <param name="stage_id">단계 이름.</param>
        <param name="start">시작 시각(perf_counter 초).</param>
        <param name="end">종료 시각(perf_counter 초).</param>
        <param name="ms">걸린 시간(밀리초).</param>
        <param name="input_summary">입력 요약 문자열.</param>
        <param name="output_summary">출력 요약 문자열.</param>
        <input>단계 측정값.</input>
        <output>없음.</output>
        <reason>output 단계를 지날 때마다 시각을 남겨 두면, 화면에 실제로 나간 프레임 기준의 FPS를 계산할 수 있다.</reason>
        """
        with self.lock:
            self._append({
                "type": "stage",
                "stage": stage_id,
                "start": start,
                "end": end,
                "ms": ms,
                "input": input_summary,
                "output": output_summary,
            })

            if stage_id == "output":
                self.output_times.append(end)

    def warn(self, message, stage_id=None):
        """
        <summary>경고 이벤트(프레임 드롭, skip으로 추론 생략 등)를 기록한다.</summary>
        <param name="message">경고 문장.</param>
        <param name="stage_id">관련 단계 이름. 없으면 None.</param>
        <input>문장과 단계 이름.</input>
        <output>없음.</output>
        <reason>오류는 아니지만 "평소와 다르게 처리된 프레임"을 눈에 띄게 남겨, 속도 설정의 영향을 로그로 확인하게 한다.</reason>
        """
        with self.lock:
            self._append({"type": "warn", "stage": stage_id, "message": message})

    def note(self, message, stage_id=None):
        """
        <summary>단계 통과는 아니지만 알려 둘 만한 일(카메라 켜짐·꺼짐 등)을 INFO 줄로 기록한다.</summary>
        <param name="message">알릴 문장.</param>
        <param name="stage_id">관련 단계 이름. 없으면 None.</param>
        <input>문장과 단계 이름.</input>
        <output>없음.</output>
        <reason>로그 콘솔에서 "언제 카메라를 껐다 켰는지"가 보여야, 그 앞뒤로 숫자가 바뀐 이유를 알 수 있다.</reason>
        """
        with self.lock:
            self._append({"type": "note", "stage": stage_id, "message": message})

    def latest_seq(self):
        """
        <summary>지금까지 기록된 마지막 이벤트 번호를 돌려준다.</summary>
        <input>없음.</input>
        <output>정수.</output>
        <reason>새로 접속한 브라우저가 어디부터 받을지 정할 때 쓴다.</reason>
        """
        with self.lock:
            return self.seq

    def snapshot(self, since_seq, max_logs):
        """
        <summary>링버퍼를 읽어 브라우저로 보낼 집계를 만든다.</summary>
        <param name="since_seq">브라우저가 이미 받은 마지막 이벤트 번호.</param>
        <param name="max_logs">한 번에 보낼 최대 로그 줄 수.</param>
        <input>이벤트 번호와 최대 줄 수.</input>
        <output>stage_ms, share, bottleneck, active, skipped, fps, frame, logs, dropped_logs, last_seq를 담은 딕셔너리.</output>
        <reason>
        lock 안에서는 링버퍼를 리스트로 복사만 하고, 계산은 lock 밖에서 한다.
        lock을 오래 잡으면 그동안 파이프라인 스레드가 기록하지 못하고 기다려야 하기 때문이다.
        </reason>
        """
        with self.lock:
            events = list(self.events)
            active = self.active
            skipped = self.inference_skipped
            frame = self.frame
            last_seq = self.seq
            times = list(self.output_times)

        stage_ms = {}
        recent = {stage: [] for stage, _ in STAGE_ORDER}

        for event in reversed(events):
            if event["type"] != "stage":
                continue

            bucket = recent.get(event["stage"])

            if bucket is not None and len(bucket) < AVERAGE_FRAMES:
                bucket.append(event["ms"])

        for stage, values in recent.items():
            stage_ms[stage] = sum(values) / len(values) if values else None

        measured = {stage: ms for stage, ms in stage_ms.items() if ms is not None}
        total_ms = sum(measured.values())
        share = {stage: (ms / total_ms * 100 if total_ms > 0 else 0) for stage, ms in measured.items()}

        if measured:
            bottleneck = max(measured, key=measured.get)
        else:
            bottleneck = None

        if len(times) >= 2 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        else:
            fps = 0.0

        new_events = [event for event in events if event["seq"] > since_seq]
        lines = []

        for event in new_events:
            lines.extend(format_event(event))

        dropped = max(0, len(lines) - max_logs)

        return {
            "stage_ms": stage_ms,
            "share": share,
            "total_ms": total_ms,
            "bottleneck": bottleneck,
            "active": active,
            "skipped": skipped,
            "fps": fps,
            "frame": frame,
            "logs": lines[-max_logs:] if max_logs > 0 else [],
            "dropped_logs": dropped,
            "last_seq": last_seq,
        }


def format_time(wall):
    """
    <summary>유닉스 시각을 "14:40:02.135" 형태로 바꾼다.</summary>
    <param name="wall">time.time() 값.</param>
    <input>실수 시각.</input>
    <output>시:분:초.밀리초 문자열.</output>
    <reason>로그끼리 시간 간격을 비교하려면 밀리초까지 보여야 한다.</reason>
    """
    return datetime.fromtimestamp(wall).strftime("%H:%M:%S.%f")[:-3]


def format_event(event):
    """
    <summary>이벤트 하나를 로그 줄(레벨, 글자)로 바꾼다.</summary>
    <param name="event">이벤트 딕셔너리.</param>
    <input>링버퍼의 이벤트.</input>
    <output>[{seq, level, text}] 목록. 단계 이벤트는 INFO와 DEBUG 두 줄, 경고는 WARN 한 줄.</output>
    <reason>
    INFO는 "어느 단계를 몇 ms에 통과했는가", DEBUG는 "데이터 모양이 어떻게 바뀌었는가"다.
    두 줄로 나눠 두면 레벨 필터로 보고 싶은 정보만 골라 볼 수 있다.
    </reason>
    """
    prefix = f"[{format_time(event['wall'])}] #frame {event['frame']:04d}"

    if event["type"] in ("warn", "note"):
        stage = event["stage"] or "-"
        level = "WARN" if event["type"] == "warn" else "INFO"
        return [{"seq": event["seq"], "level": level, "text": f"{prefix} | {stage:<11} | {event['message']}"}]

    stage = event["stage"]
    info = f"{prefix} | {stage:<11} | {event['ms']:.1f}ms"
    debug = f"{prefix} | {stage:<11} | in={event['input']} out={event['output']}"

    return [
        {"seq": event["seq"], "level": "INFO", "text": info},
        {"seq": event["seq"], "level": "DEBUG", "text": debug},
    ]


# ============================================================
# 4. 데코레이터
# ============================================================

registry = StageRegistry()
recorder = EventRecorder()


def stage(stage_id, external=False):
    """
    <summary>단계 함수를 레지스트리에 등록하고, 부를 때마다 시간과 입출력 요약을 이벤트로 기록하는 데코레이터를 만든다.</summary>
    <param name="stage_id">단계 이름. STAGE_ORDER의 이름 중 하나.</param>
    <param name="external">True면 함수가 StageReport로 시간을 직접 알려 준다(ultralytics 내부 단계용).</param>
    <input>단계 이름과 외부 측정 여부.</input>
    <output>함수를 감싸는 데코레이터.</output>
    <reason>
    functools.wraps를 써야 감싼 뒤에도 원래 함수의 이름과 docstring이 남는다. 플로우차트 설명이 바로 그 docstring이다.
    일반 단계는 perf_counter로 앞뒤 시각을 잰다.
    external 단계는 함수가 돌려준 StageReport의 시간을 그대로 기록하고, 호출한 쪽에는 결과(output)만 넘긴다.
    </reason>
    """
    def decorator(func):
        """
        <summary>함수를 등록하고 측정 코드로 감싼다.</summary>
        <param name="func">단계 함수.</param>
        <input>함수.</input>
        <output>감싼 함수.</output>
        <reason>등록은 함수를 정의하는 순간(모듈을 불러올 때) 한 번만 일어난다.</reason>
        """
        registry.register(stage_id, func, external)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            <summary>단계 함수를 실행하고 이벤트를 기록한다.</summary>
            <input>단계 함수의 인자 그대로.</input>
            <output>단계 함수의 결과(external이면 StageReport.output).</output>
            <reason>첫 번째 인자를 이 단계의 "입력"으로 보고 요약한다. 단계 함수는 첫 인자로 처리할 데이터를 받도록 맞춰 둔다.</reason>
            """
            input_summary = summarize(args[0]) if args else "-"

            if external:
                report = func(*args, **kwargs)
                end = report.start + report.ms / 1000
                recorder.record_stage(stage_id, report.start, end, report.ms, input_summary, summarize(report.output))
                return report.output

            recorder.set_active(stage_id)
            start = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                recorder.set_active(None)

            ms = (end - start) * 1000
            recorder.record_stage(stage_id, start, end, ms, input_summary, summarize(result))

            return result

        return wrapper

    return decorator
