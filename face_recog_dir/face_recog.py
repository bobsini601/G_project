# face_recog.py

import face_recognition #dlib에 있는 거 불러온것
import cv2
import camera # camera.py 불러온것
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = [] # 사진 이름 리스트

        # Load sample pictures and learn how to recognize it.
        # 'pictures' 디렉토리에는 사용자의 이미지가 있다.
        dirname = 'pictures' # 디렉토리 이름
        files = os.listdir(dirname) # listdir(): 디렉토리에 어떤 파일들이 있는지 리스트로 불러오기
        for filename in files:
            name, ext = os.path.splitext(filename) # 파일에서 이름, 확장자 분리
            if ext == '.jpg': # 확장자가 jpg 면
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename) # 파일이름을 경로에 합치기
                img = face_recognition.load_image_file(pathname) # face_recognition 이용해서 사진에서 얼굴 영역을 알아내고
                face_encoding = face_recognition.face_encodings(img)[0] # 그 얼굴 영역에서 68개 얼굴 위치의 속성 값을 분석
                self.known_face_encodings.append(face_encoding) # 얼굴 속성 값을 리스트에 저장


        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):


        frame = self.camera.get_frame() # 카메라로부터 frame 읽어서

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # frame의 크기를 1/4로 줄임

        rgb_small_frame = small_frame[:, :, ::-1] # BGR(OpenCV가 쓰는거) -> RGB(face_recognition가 쓰는거)로 바꾸기


        if self.process_this_frame: # 빠르게 하려고 두 frame당 1번씩 계산

            self.face_locations = face_recognition.face_locations(rgb_small_frame) # frame에서 얼굴 위치 추출
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations) # 얼굴 위치에서 속성 추출

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) # 사진의 얼굴 특성이랑 frame의 얼굴 특성 거리로 비교
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown" # 거리가 0.6 이상이면 다른 사람으로 인식
                if min_value < 0.6: # 0.6 이하면
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name) # 그 사진 찾아서 이름 불러오기

        self.process_this_frame = not self.process_this_frame # True면 False로, False면 True로

        # 결과 보여주는 네모 그리기
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        # frame이 나타난다.
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF # waitKey(): 키 입력을 기다리는 대기 함수. 괄호 안의 숫자는 키 입력 대기 시간으로 단위는 ms
                                    # 사실 영상을 보여주는 것이 아니라 웹캠으로 사진을 찍어서 사진을 연속으로 보여주는 것임.
                                    # 따라서 1ms마다 (키 입력을 기다리면서) 사진을 찍어서 보여주는 것. 만약 1000을 넣으면 1초마다 사진을 보여주게 되어 답답해짐

        #'ESC' 키를 누르면, while문을 빠져나옴.
        if key == 27: # 27은 ESC키를 의미
            break

    cv2.destroyAllWindows() # 창 닫음
    print('finish')
