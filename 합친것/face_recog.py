# face_recog.py
# 웹캠 동영상에 있는 얼굴을 감지하여 pictuers 디렉토리에 있는 사진의 얼굴과 비교하여 감지되는 이름을 출력하는 파일.
# pictuers 디렉토리에는 본인의 사진 1장만이 있다고 가정

import face_recognition # dlib에 있는 거 불러온것
import cv2
import camera # camera.py 불러온 것
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        self.camera = camera.VideoCamera() # camera.py의 VideoCamera 클래스

        self.known_face_encodings = [] # 사진의 얼굴 속성 값을 넣을 리스트
        self.known_face_names = [] # 얼굴 이름을 넣을 리스트
        self.is_recognized = 0 # 인식이 되었는지

        dirname = 'pictures' # 디렉토리 이름
        files = os.listdir(dirname) # listdir(): 디렉토리에 어떤 파일들이 있는지 리스트로 불러오기
        for filename in files:
            name, ext = os.path.splitext(filename) # 파일이름을 2개의 이름으로 분리(이름, 확장자명)
                                                   # 예를 들어, sein.jpg --> sein 과 .jpg
            if ext == '.jpg': # 확장자가 jpg 면
                self.known_face_names.append(name) # 얼굴 이름 리스트에 이름 추가
                pathname = os.path.join(dirname, filename) # 파일이름을 경로에 합치기
                img = face_recognition.load_image_file(pathname) # 위 경로를 통해 face_recognition에서 해당 이미지 불러오기
                face_encoding = face_recognition.face_encodings(img)[0] # 불러온 이미지에서 68개 얼굴 위치(face landmarks)의 속성 값 알아내기
                self.known_face_encodings.append(face_encoding) # 얼굴 속성 값을 리스트에 저장


        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame0(self):
        frame = self.camera.get_frame()
        return frame

    def get_frame(self):

        frame = self.camera.get_frame() # 카메라로부터 frame 읽어서
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # frame의 크기를 1/4로 줄임(계산량을 줄이기 위해)
        rgb_small_frame = small_frame[:, :, ::-1] # BGR(OpenCV가 쓰는거) -> RGB(face_recognition가 쓰는거)로 바꾸기


        if self.process_this_frame: # 두 frame당 1번씩 계산(계산량을 줄이기 위해)

            self.face_locations = face_recognition.face_locations(rgb_small_frame) # frame에서 얼굴 위치 추출
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations) # 얼굴 위치에서 face landmark 추출

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) # 사진의 face landmark와 frame의 face landmark를 거리로 비교
                min_value = min(distances) # 이거 없애도 될듯

                name = "Unknown" # 거리가 0.6 이상이면 다른 사람으로 인식
                if min_value < 0.6: # 0.6 이하면
                    self.is_recognized += 1
                    #print("your face is recognized!")
                    index = np.argmin(distances)
                    name = self.known_face_names[index] # 사진의 이름 불러오기
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

    # def get_jpg_bytes(self):
    #     frame = self.get_frame()
    #     # We are using Motion JPEG, but OpenCV defaults to capture raw images,
    #     # so we must encode it into JPEG in order to correctly display the
    #     # video stream.
    #     ret, jpg = cv2.imencode('.jpg', frame)
    #     return jpg.tobytes()


# if __name__ == '__main__':
#     face_recog = FaceRecog()
#     print(face_recog.known_face_names)
#
#     while True:
#         if face_recog.is_recognized < 5:
#             frame = face_recog.get_frame()
#             if face_recog.is_recognized == 5:
#                 print("your face is recognized!")
#         else:
#             frame = face_recog.get_frame0()
#
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF # waitKey(): 키 입력을 기다리는 대기 함수. 괄호 안의 숫자는 키 입력 대기 시간으로 단위는 ms
#                                     # 사실 영상을 보여주는 것이 아니라 웹캠으로 사진을 찍어서 사진을 연속으로 보여주는 것임.
#                                     # 따라서 1ms마다 (키 입력을 기다리면서) 사진을 찍어서 보여주는 것. 만약 1000을 넣으면 1초마다 사진을 보여주게 되어 답답해짐
#
#         if key == 27: # 27은 ESC키를 의미
#             break # ESC키 누르면 while문 탈출
#
#     cv2.destroyAllWindows() # 창 닫음
#     print('finish')
face_recog = FaceRecog()
print(face_recog.known_face_names)

while True:
    if face_recog.is_recognized < 5:
        frame = face_recog.get_frame()
        if face_recog.is_recognized == 5:
            print("your face is recognized!")
    else:
        frame = face_recog.get_frame0()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF # waitKey(): 키 입력을 기다리는 대기 함수. 괄호 안의 숫자는 키 입력 대기 시간으로 단위는 ms
                                    # 사실 영상을 보여주는 것이 아니라 웹캠으로 사진을 찍어서 사진을 연속으로 보여주는 것임.
                                    # 따라서 1ms마다 (키 입력을 기다리면서) 사진을 찍어서 보여주는 것. 만약 1000을 넣으면 1초마다 사진을 보여주게 되어 답답해짐

    if key == 27: # 27은 ESC키를 의미
        break # ESC키 누르면 while문 탈출

cv2.destroyAllWindows() # 창 닫음
print('finish')