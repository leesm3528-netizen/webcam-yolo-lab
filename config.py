"""
<summary>웹캠 실시간 YOLO 실습 전 단계(step1~step7)가 함께 쓰는 설정 파일.</summary>
<reason>
카메라 번호, 모델, 장치 같은 값을 파일마다 따로 적으면 바꿀 때 실수가 생긴다.
한 곳에 모아 두면 학생은 이 파일만 고쳐서 모든 단계의 동작을 바꿀 수 있다.
</reason>
"""


# ------------------------------------------------------------
# 연산 장치 설정
# ------------------------------------------------------------
# "cpu"    : GPU가 없는 노트북 (기본값)
# "cuda:0" : NVIDIA GPU
# "mps"    : Apple Silicon(M1/M2/M3) 맥
DEVICE = "cpu"


# ------------------------------------------------------------
# 카메라 설정
# ------------------------------------------------------------
# 노트북 내장 카메라는 보통 0번, 외장 USB 카메라는 1번 이상이다.
CAMERA_INDEX = 0

# 요청할 해상도. 카메라가 지원하지 않으면 가장 가까운 값으로 열린다.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# True면 거울처럼 좌우를 뒤집어 보여 준다. (2단계에서 원리를 다룬다)
FLIP_HORIZONTAL = True


# ------------------------------------------------------------
# YOLO 모델 설정
# ------------------------------------------------------------
# 처음 실행할 때 자동으로 내려받는다. CPU에서는 가장 작은 n 모델을 권장한다.
# 내려받기에 실패하거나 ultralytics 버전이 낮아 열리지 않으면 "yolo11n.pt"로 바꾼다.
MODEL_PATH = "yolo26n.pt"

# 실패했을 때 안내할 대체 모델
FALLBACK_MODEL_PATH = "yolo11n.pt"

# 이 값보다 신뢰도가 낮은 검출 결과는 버린다. (0.0 ~ 1.0)
CONF_THRESHOLD = 0.5

# 모델에 넣는 이미지 크기. 작을수록 빠르지만 작은 물체를 놓치기 쉽다.
IMG_SIZE = 640


# ------------------------------------------------------------
# 화면 표시 설정
# ------------------------------------------------------------
WINDOW_NAME = "YOLO Webcam Lab"

# 이 키를 누르면 프로그램을 종료한다.
QUIT_KEY = "q"


# ------------------------------------------------------------
# Flask 웹 스트리밍 설정
# ------------------------------------------------------------
# "127.0.0.1"이면 내 컴퓨터에서만, "0.0.0.0"이면 같은 네트워크의 다른 기기에서도 접속된다.
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

# 브라우저로 보내는 JPEG 화질. (0 ~ 100)
JPEG_QUALITY = 80


def resolve_device(requested=DEVICE):
    """
    <summary>설정한 장치를 실제로 쓸 수 있는지 확인하고, 안 되면 CPU로 바꿔 돌려준다.</summary>
    <param name="requested">config.py에 적은 장치 이름. "cpu", "cuda:0", "mps" 중 하나.</param>
    <input>장치 이름 문자열.</input>
    <output>실제로 사용할 장치 이름 문자열.</output>
    <reason>
    학생이 GPU가 없는 노트북에서 DEVICE를 "cuda:0"으로 바꿔도 프로그램이 멈추지 않게 한다.
    무엇이 문제인지 알려 주고 CPU로 계속 진행해야 실습 흐름이 끊기지 않는다.
    </reason>
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        print("[안내] PyTorch를 불러올 수 없어 CPU로 실행합니다.")
        return "cpu"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested

        print(f"[안내] '{requested}'를 사용할 수 없습니다. (CUDA용 PyTorch 또는 NVIDIA GPU 없음)")
        print("[안내] CPU로 실행합니다.")
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"

        print("[안내] 'mps'를 사용할 수 없습니다. (Apple Silicon 맥이 아님)")
        print("[안내] CPU로 실행합니다.")
        return "cpu"

    print(f"[안내] 알 수 없는 장치 이름 '{requested}'입니다. CPU로 실행합니다.")
    return "cpu"
