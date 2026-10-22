"""
<summary>1단계: 웹캠을 열어 실시간 영상을 창에 띄운다.</summary>
<reason>
YOLO를 붙이기 전에 "카메라가 열리는가, 프레임이 들어오는가"부터 확인해야 한다.
여기서 문제가 생기면 이후 모든 단계가 동작하지 않으므로 가장 먼저 점검한다.
</reason>
"""

import sys

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
    <reason>
    Windows에서는 CAP_DSHOW(DirectShow)로 열어야 카메라가 빨리 켜지고 경고가 적다.
    다른 운영체제에서는 OpenCV가 알아서 고르도록 기본값을 쓴다.
    </reason>
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
    <reason>
    "카메라를 열 수 없습니다"만 보여 주면 학생이 무엇을 고쳐야 할지 모른다.
    자주 겪는 원인을 순서대로 알려 주면 스스로 해결할 수 있다.
    </reason>
    """
    print(f"[오류] {index}번 카메라를 열 수 없습니다.")
    print("다음을 확인하세요.")
    print("  1. 줌, 팀즈, 카메라 앱 등 다른 프로그램이 카메라를 쓰고 있지 않은가?")
    print("  2. Windows 설정 > 개인 정보 및 보안 > 카메라 에서 '데스크톱 앱의 카메라 액세스'가 켜져 있는가?")
    print("  3. 노트북의 카메라 차단 스위치나 Fn 단축키로 카메라가 꺼져 있지 않은가?")
    print("  4. 외장 카메라라면 config.py의 CAMERA_INDEX를 1, 2로 바꿔 보았는가?")


def run():
    """
    <summary>웹캠 영상을 읽어 창에 계속 표시하고, 종료 키를 누르면 끝낸다.</summary>
    <input>config.py의 카메라 번호, 해상도, 창 이름, 종료 키.</input>
    <output>프로그램 종료 코드. 정상 종료는 0, 카메라 오류는 1.</output>
    <reason>
    try/finally로 감싸서 어떤 이유로 끝나든 카메라와 창을 반드시 해제한다.
    해제하지 않으면 다음 실행 때 카메라가 "사용 중"으로 남아 열리지 않을 수 있다.
    </reason>
    """
    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    if cap is None:
        print_camera_help(config.CAMERA_INDEX)
        return 1

    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[정보] 카메라가 열렸습니다. 해상도: {real_width} x {real_height}")
    print(f"[정보] 창을 선택한 상태에서 '{config.QUIT_KEY}' 키를 누르면 종료합니다.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[오류] 카메라에서 프레임을 읽지 못했습니다.")
                print("카메라 연결이 끊겼거나 다른 프로그램이 카메라를 가져갔을 수 있습니다.")
                break

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("[정보] 종료 키를 눌러 프로그램을 끝냅니다.")
                break

            # 창의 X 버튼을 눌러 닫은 경우에도 반복을 끝낸다.
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
#      python step1_webcam.py
# 3. 영상 창을 클릭한 뒤 q 키를 누르면 종료한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# - 콘솔에 출력된 실제 해상도가 config.py에 적은 640 x 480과 같은가?
#   카메라가 지원하지 않는 해상도는 가장 가까운 값으로 바뀐다.
# - 영상이 거울처럼 좌우 반전되어 보이는가, 아닌가? (2단계에서 다룬다)
# - 다른 앱에서 카메라를 켜 둔 채 실행하면 어떤 메시지가 나오는가?
# - 종료한 뒤 바로 다시 실행해도 카메라가 정상적으로 열리는가?
#   (finally에서 cap.release()를 해 주었기 때문이다)
