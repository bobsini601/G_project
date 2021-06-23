# camera.py
# 웹캠의 동영상을 모니터로 출력하는 파일. 카메라가 정상적으로 동작하는지 검사하기 위한 것.

import cv2

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        return frame


if __name__ == '__main__':
    cam = VideoCamera()
    while True:
        frame = cam.get_frame()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print('finish')
