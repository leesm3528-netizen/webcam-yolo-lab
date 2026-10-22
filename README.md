# 웹캠 실시간 YOLO 객체 인식 실습 (8단계)

> 노트북 웹캠 영상을 OpenCV로 읽고, Flask로 브라우저에 띄우고, YOLO로 실시간 객체 인식을 한 뒤,
> 마지막에는 **"한 프레임이 어떤 단계를 지나가는지"** 를 플로우차트와 로그로 눈으로 확인하는 실습입니다.

| 항목 | 내용 |
|---|---|
| 실습일 | 2026-10-22 |
| 언어 · 라이브러리 | Python 3.10+ · OpenCV · Ultralytics YOLO · Flask · NumPy · PyTorch |
| 모델 | `yolo26n.pt` (기본) · `yolo11n.pt` (대체) |
| 실행 환경 | GPU 없는 노트북(CPU) 기본. `config.py`의 `DEVICE`만 바꾸면 GPU(`cuda:0`)·Apple Silicon(`mps`)로 동작 |
| 테스트 환경 | Windows 11 · GTX 1650 Ti 4GB · torch 2.14.1+cu132 · ultralytics 8.4.183 · opencv-python 5.0.0 |

---

## 목차

1. [학습 목표](#1-학습-목표)
2. [환경 준비](#2-환경-준비)
3. [전체 흐름 한눈에 보기](#3-전체-흐름-한눈에-보기)
4. [단계별 실습](#4-단계별-실습)
5. [측정 결과 예시](#5-측정-결과-예시)
6. [단축키 모음](#6-단축키-모음)
7. [트러블슈팅](#7-트러블슈팅)
8. [오인식 개선 방법 (지갑이 cell phone으로 잡힐 때)](#8-오인식-개선-방법)
9. [복습 퀴즈](#9-복습-퀴즈)
10. [파일 구조](#10-파일-구조)

---

## 1. 학습 목표

이 실습을 마치면 다음을 설명하고 직접 해 볼 수 있습니다.

- 웹캠을 열고, 프레임을 읽고, **반드시 해제**하는 기본 흐름
- **동영상 = 연속된 이미지 한 장의 모음**이라는 사실 (이미지 = numpy 배열)
- OpenCV의 **BGR 색 순서**와 **(x, y) 좌표 vs 배열 `[y, x]` 인덱스**의 차이
- Flask로 영상을 브라우저에 보내는 **MJPEG 스트리밍**(`multipart/x-mixed-replace`)과, 카메라를 스레드 + lock으로 공유하는 이유
- YOLO 결과(`boxes.xyxy`, `conf`, `cls`)를 꺼내 **직접 상자를 그리는 법**
- `imgsz`, `skip`, CPU/GPU에 따라 **속도(FPS)와 정확도가 어떻게 맞바뀌는지**
- 한 프레임을 `capture → preprocess → inference → postprocess → visualize → output` 으로 나눠 **병목을 숫자로 찾는 법**

---

## 2. 환경 준비

### 2-1. 가상환경 만들기

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2-2. 라이브러리 설치

```powershell
pip install -r requirements.txt
```

### 2-3. (선택) NVIDIA GPU를 쓰는 경우

`pip install torch`만 하면 **CPU 버전**이 깔립니다. GPU를 쓰려면 [pytorch.org](https://pytorch.org/get-started/locally/) 설치 선택기에서
본인 CUDA 버전에 맞는 명령을 복사해 실행합니다.

```powershell
# 예시 (이번 실습 환경: CUDA 13.2용)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 확인: True 와 GPU 이름이 나오면 성공
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **알아 두기**
> - CUDA Toolkit을 따로 설치할 필요는 없습니다. PyTorch 설치 파일 안에 CUDA 런타임이 들어 있습니다.
> - 그래픽 드라이버가 해당 CUDA 버전을 지원하는지만 확인하면 됩니다 (`nvidia-smi` 오른쪽 위 `CUDA Version`).
> - 나중에 주소 없이 `pip install torch`를 다시 실행하면 CPU 버전으로 돌아갑니다.

### 2-4. 설정 파일 `config.py`

모든 단계가 이 파일 하나의 값을 함께 씁니다. **학생은 이 파일만 고치면 됩니다.**

| 설정 | 기본값 | 설명 |
|---|---|---|
| `DEVICE` | `"cpu"` | `"cuda:0"`, `"mps"`로 변경 가능. 쓸 수 없으면 이유를 알려 주고 CPU로 계속 실행 |
| `CAMERA_INDEX` | `0` | 내장 카메라 0, 외장 USB 카메라는 1 이상 |
| `FRAME_WIDTH/HEIGHT` | `640 × 480` | 요청 해상도 |
| `MODEL_PATH` | `"yolo26n.pt"` | 첫 실행 시 자동 다운로드. 문제 있으면 `"yolo11n.pt"` |
| `CONF_THRESHOLD` | `0.5` | 이보다 낮은 신뢰도의 결과는 버림 |
| `IMG_SIZE` | `640` | 모델 입력 크기. 작을수록 빠르지만 작은 물체를 놓침 |
| `FLASK_HOST/PORT` | `127.0.0.1:5000` | 웹 서버 주소 |

---

## 3. 전체 흐름 한눈에 보기

```
STEP 1  웹캠 열기 ──▶ STEP 2  브라우저로 스트리밍 ──▶ STEP 3  촬영 버튼으로 jpg 저장
                                                              │ (저장한 사진이 다음 단계 재료)
                                                              ▼
STEP 6  사진 한 장에 YOLO ◀── STEP 5  실시간 영상 처리 ◀── STEP 4  OpenCV 이미지 기초
   │
   ▼
STEP 7  실시간 YOLO (창 / 웹 모드) ──▶ STEP 8  파이프라인 대시보드 (플로우차트 + 로그)
```

모든 함수·클래스에는 XML 형식 docstring이 들어 있습니다. 코드를 읽을 때 이 태그부터 보세요.

```python
"""
<summary>이 함수가 하는 일</summary>
<param name="frame">입력 매개변수 설명</param>
<input>무엇을 받는지</input>
<output>무엇을 돌려주는지</output>
<reason>왜 이 방식을 썼는지</reason>
"""
```

STEP 8에서는 이 docstring을 서버가 직접 읽어 **플로우차트 노드의 설명**으로 보여 줍니다.

---

## 4. 단계별 실습

모든 단계는 **독립 실행**됩니다. 가상환경을 켠 뒤 `python 파일이름.py`로 실행합니다.

### STEP 1. 웹캠 열기 — `step1_webcam.py`

```powershell
python step1_webcam.py        # 창을 한 번 클릭한 뒤 q 로 종료
```

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 0번 카메라 (Windows는 CAP_DSHOW가 빠름)
if not cap.isOpened():                      # 열렸는지 꼭 확인
    print("카메라를 열 수 없습니다")
ok, frame = cap.read()                      # 프레임 1장 (반복문 안에서 계속 호출)
cap.release()                               # 다 쓰면 반드시 해제 (finally에서)
```

**확인 포인트**
- `release()` 없이 끝내면 다음 실행 때 카메라가 "사용 중"으로 남을 수 있다.
- 카메라가 안 열리면 프로그램이 확인할 항목(다른 앱 사용 중, Windows 권한, 차단 스위치, 카메라 번호)을 한국어로 알려 준다.

### STEP 2. Flask MJPEG 스트리밍 — `step2_server.py`

```powershell
python step2_server.py        # 브라우저에서 http://127.0.0.1:5000
```

- `Camera` 클래스가 **별도 스레드**에서 카메라를 계속 읽고, **최신 프레임 1장만 lock으로 보호**해 공유합니다.
- `/video_feed`는 `multipart/x-mixed-replace; boundary=frame` 응답으로 JPEG를 끝없이 이어 보냅니다.
- 서버가 종료되면 `atexit`로 카메라를 해제합니다.

**확인 포인트**
- 요청마다 카메라를 새로 열면 왜 안 될까? → 장치는 하나인데 여러 요청이 동시에 열려고 하면 충돌한다.
- lock이 없으면? → 한 스레드가 프레임을 쓰는 도중 다른 스레드가 읽어 프레임과 번호가 어긋날 수 있다.
- 개발자 도구(Network 탭)에서 `/video_feed`의 Content-Type을 확인해 보자.

### STEP 3. 촬영 버튼 — `step3_capture.py`

```powershell
python step3_capture.py       # 촬영 버튼 또는 Space
```

- "촬영" → `POST /capture` → 최신 프레임을 `captures/YYYYMMDD_HHMMSS.jpg`로 저장 → 파일명을 JSON으로 반환
- 페이지는 `fetch`로 응답을 받아 **새로고침 없이** 캡처 이미지와 최근 10장 썸네일을 표시
- 같은 초에 두 번 찍으면 `_2.jpg`로 저장되어 덮어쓰지 않음

**보안 포인트 — 경로 조작(`../`) 방지 2중 장치**
1. 파일명이 `YYYYMMDD_HHMMSS(_n).jpg` 규칙과 맞지 않으면 파일을 찾지도 않고 404 (위험한 문자를 하나씩 막는 대신 **허용된 모양만 통과**)
2. `send_from_directory`로 `captures/` 폴더 밖은 열 수 없게 함

**확인 포인트**: 동영상은 연속된 이미지 한 장의 모음이다. 여기서 저장한 jpg가 STEP 4·6의 실습 재료가 된다.

### STEP 4. OpenCV 이미지 기초 — `step4_opencv_basics.py`

```powershell
python step4_opencv_basics.py    # captures/가 비어 있으면 웹캠으로 1장 찍어서 사용
```

- `cv2.imread` 후 `shape (480, 640, 3)`, `dtype uint8`, 값 범위 `0~255`, 채널별 최소/최대/평균 출력
- BGR · RGB · 그레이스케일을 `cv2.hconcat`으로 나란히 비교 (크기와 채널 수를 맞춰야 이어 붙일 수 있음)
- 같은 픽셀에서 `bgr=[48,110,133]` → `rgb=[133,110,48]` 처럼 첫 값과 끝 값이 바뀐다
- 좌표계: `cv2.circle(img, (x=200, y=80))` 와 `img[200, 80]` 이 **다른 위치**를 가리킨다

**확인 포인트**
- OpenCV가 BGR을 쓰는 이유: 초기 카메라·BMP 형식의 역사적 관례
- YOLO 입력 전에 색 순서가 중요한 이유: 순서가 틀려도 **오류 없이 결과만 틀려진다**
- 그레이를 3채널로 바꾸지 않고 `hconcat`하면 `(-215:Assertion failed)` 오류가 난다 → 직접 확인해 보자

### STEP 5. 실시간 영상 처리 — `step5_opencv_practice.py`

```powershell
python step5_opencv_practice.py
```

| 키 | 모드 |
|---|---|
| `1` | 원본 |
| `2` | 그레이 |
| `3` | 가우시안 블러 |
| `4` | Canny 엣지 |
| `5` | 이진화(threshold) |
| `6` | HSV 색상 마스크 (트랙바로 H/S/V 범위 조절) |
| `s` / `q` | 저장 / 종료 |

- 모든 모드에 `cv2.rectangle` + `cv2.putText`로 모드명과 FPS를 그립니다. **이 그리기 함수가 STEP 6~7의 박스 시각화에 그대로 쓰입니다.**
- FPS는 `time.perf_counter`로 이동평균을 계산합니다.
- 화면의 `Capacity` = 1000 ÷ 처리 시간(ms). 카메라가 초당 15장만 주기 때문에 FPS는 모드와 상관없이 비슷하게 보이므로, **처리만 따졌을 때 감당할 수 있는 프레임 수**를 따로 보여 줍니다.
- 빨간색 검출 팁: 색상환 양 끝에 걸쳐 있으므로 `H min=170, H max=10` 으로 두 구간을 합친다.

**확인 포인트**: 블러 커널을 21 → 51로 키우면 처리 시간이 약 3배 늘어난다. 연산량이 실시간성에 미치는 영향을 체감해 보자.

### STEP 6. 사진 한 장에 YOLO — `step6_yolo_image.py`

```powershell
python step6_yolo_image.py
```

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
results = model.predict(source=img, imgsz=640, conf=0.4, device="cpu")
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]            # 좌표
    name = model.names[int(box.cls[0])]     # 클래스 이름
    score = float(box.conf[0])              # 신뢰도
```

- 직접 그린 결과와 `results[0].plot()` 결과를 나란히 비교 → 상자 위치가 정확히 일치
- `imgsz` 320 / 416 / 640 속도·탐지 수 비교 함수 포함
- **첫 추론은 워밍업**(CPU 약 3.5초, 이후 60ms)이라 측정에서 제외
- GPU일 때 FP16 사용. 단, FP16과 FP32의 탐지 수가 다르면 자동으로 FP32로 전환 (아래 트러블슈팅 참고)

**확인 포인트**: `results[0].speed`의 전처리·추론·후처리 비중 → CPU에서는 추론이 약 94%

### STEP 7. 실시간 YOLO (최종) — `step7_realtime_yolo.py`

```powershell
python step7_realtime_yolo.py                          # cv2 창 모드 (CPU 기본: imgsz 320, skip 2)
python step7_realtime_yolo.py --mode web               # 브라우저 모드 http://127.0.0.1:5000
python step7_realtime_yolo.py --device cuda:0          # GPU: imgsz 640, skip 1, FP16 자동
python step7_realtime_yolo.py --classes person cup     # 원하는 클래스만
```

| 옵션 | 설명 |
|---|---|
| `--mode window\|web` | cv2 창 / Flask 웹 (`/detect_feed`, 촬영 시 탐지 결과 프레임 저장) |
| `--device cpu\|cuda:0\|mps` | 추론 장치 |
| `--imgsz` | 모델 입력 크기 (생략 시 CPU 320, GPU 640) |
| `--conf` | 신뢰도 기준 |
| `--skip N` | N프레임마다 한 번만 추론, 나머지 프레임은 **직전 박스 재사용** |
| `--model` | 모델 파일 |
| `--openvino` | `yolo26n_openvino_model` 폴더가 있으면 OpenVINO로 로드 (Intel CPU 가속) |
| `--classes` | 클래스 필터. 화면에 클래스별 개수 집계 표시 |

화면에는 FPS, 추론 ms, 탐지 개수, 사용 장치, 그리고 박스가 새 추론 결과인지(`[new]`) 재사용인지가 표시됩니다.

**확인 포인트**: CPU vs GPU, `imgsz`, `skip` 값을 바꿔 가며 FPS를 표로 기록해 보자. ([5장 측정 예시](#5-측정-결과-예시) 참고)

### STEP 8. 파이프라인 대시보드 — `step8_pipeline_dashboard.py` + `pipeline.py`

```powershell
python step8_pipeline_dashboard.py
python step8_pipeline_dashboard.py --device cuda:0
python step8_pipeline_dashboard.py --imgsz 640 --skip 1
```

브라우저에서 `http://127.0.0.1:5000` 에 접속합니다.

```
┌──────────── 상단 바: FPS · 장치 · imgsz · conf · skip · [카메라 끄기] ────────────┐
│                         │  플로우차트 (capture → … → output)                      │
│  탐지 결과 스트림        │  · 현재 단계 테두리 강조 · 노드별 평균 ms · 병목 표시     │
│  + 촬영 버튼             │  · 노드 클릭 → docstring 설명 · skip 프레임은 inference 흐림│
│                         ├─────────────────────────────────────────────────────────┤
│                         │  로그 콘솔 (일시정지 · INFO/DEBUG/WARN 필터)              │
└─────────────────────────┴─────────────────────────────────────────────────────────┘
[14:40:02.135] #frame 0421 | inference | 38.2ms | in=(1,3,320,320) out=5 boxes
```

**어떻게 만들었나**

| 구성 | 내용 |
|---|---|
| `@stage("inference")` 데코레이터 | 측정 코드를 본래 처리 코드에 섞지 않고 한 줄로 덧붙임. 시작/종료 시각, 소요 ms, 프레임 번호, 입출력 요약을 이벤트로 기록 |
| 숨은 3단계 분리 | `model()` 한 번 안에 전처리·추론·후처리가 숨어 있으므로 `results[0].speed` 값을 각각의 이벤트로 나눠 기록 |
| 링버퍼 | `collections.deque(maxlen=500)` + lock으로 스레드 안전하게 보관 |
| 플로우차트 설명 | `inspect.getdoc`으로 docstring을 읽고 `xml.etree.ElementTree`로 파싱 → `GET /api/pipeline` |
| 실시간 전송 | **SSE**(`GET /events`)로 200ms마다 집계해 푸시. 단방향이고 Flask 기본 기능이라 WebSocket 라이브러리가 필요 없음 |
| 로그 레벨 | INFO = 단계 통과 · DEBUG = 입출력 shape · WARN = 프레임 드롭, skip으로 추론 생략 |
| 카메라 on/off | `http://127.0.0.1:5000/camera` 또는 상단 바 버튼. 끄면 장치가 실제로 해제되어 웹캠 불이 꺼짐 |

**확인 포인트**
1. CPU에서 inference는 전체 시간의 몇 %인가? (테스트 환경에서는 약 88%) `--device cuda:0`이면 다음 병목은 어디인가?
2. `--imgsz 320`과 `640`을 비교하면 어느 노드의 숫자가 변하는가? visualize·output은 왜 거의 그대로인가? (힌트: 모델 입력이 아니라 카메라 원본 480×640 기준으로 일한다)
3. `--skip` 값을 키우면 로그의 WARN 빈도는 어떻게 바뀌는가?

---

## 5. 측정 결과 예시

테스트 노트북(GTX 1650 Ti) 기준입니다. 여러분의 숫자는 다를 수 있으니 **직접 재서 표를 채워 보세요.**

**STEP 5 — 모드별 처리 시간 (카메라 15 FPS)**

| 모드 | 처리 시간 |
|---|---|
| 원본 | 0.22 ms |
| 그레이 | 0.19 ms |
| 가우시안 블러 (21×21) | 1.68 ms |
| Canny 엣지 | 1.72 ms |
| 이진화 | 0.30 ms |
| HSV 마스크 | 1.99 ms |

**STEP 6 — imgsz별 한 장 처리 시간**

| imgsz | CPU | GPU |
|---|---|---|
| 320 | 30.7 ms | 19.0 ms |
| 416 | 48.2 ms | 15.4 ms |
| 640 | 86.1 ms | 18.0 ms |

→ CPU는 imgsz에 민감하고, GPU는 크기가 커져도 거의 차이가 없다.

**STEP 7 — 설정별 FPS**

| 설정 | FPS | 추론 |
|---|---|---|
| CPU 기본 (imgsz 320, skip 2) | 17.9 | 27 ms |
| CPU `--skip 1` | 15.5 | 22 ms |
| CPU `--imgsz 640 --skip 1` | 10.0 | 57 ms |
| `--device cuda:0` (640, skip 1, FP16) | 12.4 | 25 ms |

**yolo26n vs yolo11n** — 같은 사진(bus, zidane, 캡처 사진)에서 찾은 물체가 완전히 같고 신뢰도도 비슷했습니다. 그래서 기본 모델은 `yolo26n`으로 유지했습니다.

---

## 6. 단축키 모음

키를 누르기 전에 **영상 창을 한 번 클릭**해야 입력이 들어갑니다. 모든 창은 X 버튼으로 닫아도 종료됩니다.

| 파일 | 키 |
|---|---|
| step1_webcam | `q` 종료 |
| step5_opencv_practice | `1`~`6` 모드, `s` 저장, `q` 종료 |
| step4_opencv_basics, step6_yolo_image | 아무 키 = 종료 |
| step7_realtime_yolo (창 모드) | `s` 탐지 결과 저장, `q` 종료 |
| 웹 페이지 (step3, step7 web, step8) | `Space` 촬영 |
| step8 플로우차트 | `Tab` 노드 이동, `Enter` 설명 보기 |
| 웹 서버 종료 | 터미널에서 `Ctrl+C` |

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `torch.cuda.is_available()`가 `False` | CPU용 torch(`+cpu`)가 설치됨 | `--index-url https://download.pytorch.org/whl/cu1xx`를 붙여 torch·torchvision 재설치 |
| 카메라가 안 열림 | 다른 프로그램(줌, 팀즈, 다른 실습 코드)이 사용 중 | 그 프로그램을 종료. Windows 설정 → 개인 정보 및 보안 → 카메라 권한 확인 |
| `half=True` 경고가 매번 뜸 | ultralytics 8.4에서 `half`가 deprecated | 코드에서 지원 여부를 확인해 `quantize=16`으로 전달, 구버전이면 `half=True` |
| GPU FP16에서 imgsz 416일 때 아무것도 안 잡힘 | GTX 1650 Ti의 FP16 수치 문제 (conf 0.01에서도 0개) | 실제 이미지로 FP16/FP32 탐지 수를 비교해 다르면 자동으로 FP32 사용 |
| 글자 테두리가 두 겹으로 어긋나 보임 | OpenCV 5에서 굵기가 다르면 글자 폭이 달라짐 | 글자 뒤에 어두운 띠를 깔고 글자는 한 번만 그림 |
| `hconcat`에서 `(-215:Assertion failed)` | 이어 붙일 이미지의 크기·채널 수가 다름 | 그레이를 `cv2.COLOR_GRAY2BGR`로 3채널 변환, 크기 맞추기 |
| GPU 메모리 부족 (학습 시) | 4GB VRAM | n/s 모델 사용, batch를 8~16으로 낮추기 |

---

## 8. 오인식 개선 방법

> 지갑이 `cell phone`으로 잡히는 이유: yolo26n은 **COCO 80개 클래스**만 배웠고 거기에 wallet이 없습니다.
> 모르는 물체를 보면 비슷하게 생긴 아는 물체로 답합니다. 설정만으로 "지갑"이라고 부르게 할 수는 없습니다.

| 방법 | 예시 | 장단점 |
|---|---|---|
| 1. 오탐 줄이기 | `--conf 0.6`, `--classes person cup` | 지금 바로 가능. 지갑을 인식하지는 못함 |
| 2. 더 큰 모델 | `--model yolo26s.pt --imgsz 640` | 헷갈림이 줄지만 CPU에서 2~3배 느림 |
| 3. 글자로 클래스 지정 (YOLOE / YOLO-World) | `model.set_classes(["wallet", "cell phone"])` | 학습 없이 지갑 탐지. 무겁고 정확도는 학습 모델보다 낮을 수 있음 |
| 4. 직접 학습 | STEP 3으로 지갑 사진 100~300장 → 라벨링(Roboflow, CVAT) → `model.train(data="wallet.yaml", epochs=50)` | 가장 확실함. 데이터 준비에 시간이 듦 |

```python
from ultralytics import YOLOE
model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(["wallet", "cell phone", "cup"])
```

---

## 9. 복습 퀴즈

<details>
<summary>Q1. 웹 서버에서 요청마다 <code>cv2.VideoCapture(0)</code>을 새로 열면 어떤 문제가 생길까요?</summary>

카메라 장치는 하나인데 여러 요청이 동시에 열려고 해서 충돌합니다. 그래서 카메라는 한 번만 열고, 스레드가 읽은 최신 프레임을 lock으로 보호해 공유합니다.
</details>

<details>
<summary>Q2. <code>img[200, 80]</code>은 x=200, y=80 위치의 픽셀일까요?</summary>

아닙니다. numpy 배열 인덱스는 `[행(y), 열(x)]` 순서라서 y=200, x=80 위치입니다. 반면 `cv2.circle`, `cv2.rectangle` 같은 그리기 함수는 `(x, y)` 순서를 씁니다.
</details>

<details>
<summary>Q3. 색 순서(BGR/RGB)를 틀리게 넣으면 YOLO는 오류를 낼까요?</summary>

오류는 나지 않고 결과만 나빠집니다. 그래서 더 찾기 어렵습니다. ultralytics는 numpy 입력을 BGR로 간주합니다.
</details>

<details>
<summary>Q4. <code>--skip 2</code>는 무엇을 하고, 왜 CPU에서 기본값일까요?</summary>

2프레임마다 한 번만 추론하고, 건너뛴 프레임에는 직전 박스를 재사용합니다. CPU에서는 추론이 전체 시간의 대부분(약 90%)을 차지하므로 추론 횟수를 줄이면 FPS가 크게 오릅니다.
</details>

<details>
<summary>Q5. STEP 8에서 WebSocket 대신 SSE를 고른 이유는?</summary>

서버 → 브라우저 단방향 전송이면 충분하고, Flask가 기본으로 지원해 추가 라이브러리가 필요 없기 때문입니다.
</details>

<details>
<summary>Q6. 첫 번째 추론 시간을 측정에서 빼는 이유는?</summary>

첫 추론에는 모델 초기화, 메모리 할당 같은 준비 작업(워밍업)이 포함되어 훨씬 느립니다(CPU 약 3.5초 vs 이후 60ms). 포함하면 평균이 왜곡됩니다.
</details>

<details>
<summary>Q7. imgsz를 320 → 640으로 바꿨는데 visualize, output 노드의 ms는 거의 그대로입니다. 왜일까요?</summary>

이 두 단계는 모델 입력이 아니라 카메라 원본 크기(480×640) 이미지에 상자를 그리고 JPEG로 인코딩하기 때문입니다.
</details>

---

## 10. 파일 구조

```
.
├── config.py                    # 모든 단계 공통 설정 (DEVICE, 카메라, 모델 등)
├── requirements.txt
│
├── step1_webcam.py              # STEP 1 웹캠 열기
├── step2_server.py              # STEP 2 Flask MJPEG 스트리밍
├── step3_capture.py             # STEP 3 촬영 버튼, captures/ 저장
├── step4_opencv_basics.py       # STEP 4 배열 정보, BGR/RGB, 좌표계
├── step5_opencv_practice.py     # STEP 5 키보드 모드 전환 영상 처리
├── step6_yolo_image.py          # STEP 6 사진 한 장 YOLO, imgsz 비교
├── step7_realtime_yolo.py       # STEP 7 실시간 YOLO (창 / 웹)
├── step8_pipeline_dashboard.py  # STEP 8 파이프라인 대시보드 서버
├── pipeline.py                  # STEP 8 @stage 데코레이터, 이벤트 링버퍼, docstring 파싱
│
├── templates/                   # index.html, step3_capture.html, step7_realtime.html, dashboard.html, camera.html
├── static/                      # dashboard.css, dashboard.js
│
└── (참고용) 처음 제안했던 7단계 구성안 파일
    ├── step2_frame_process.py   # 좌우 반전·흑백·크기 조절·FPS
    ├── step3_yolo_image.py      # 샘플 이미지 YOLO 결과 구조 보기
    ├── step4_draw_boxes.py      # 박스·라벨 직접 그리기 vs result.plot()
    ├── step5_webcam_yolo.py     # 웹캠 + YOLO 기본형
    └── step6_flask_stream.py    # Flask 스트리밍 기본형
```

> `captures/`, `outputs/`(촬영 사진·결과 이미지)와 모델 가중치(`*.pt`)는 저장소에 올리지 않았습니다.
> 모델은 첫 실행 때 자동으로 내려받고, 폴더는 실행 중 자동으로 만들어집니다.
