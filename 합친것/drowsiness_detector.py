# picture 디렉토리에는 본인의 사진 1장만이 있다고 가정

import os
import numpy as np
import time
import timeit
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
from threading import Timer
import face_recognition  # dlib에 있는 거 불러온것
import camera  # camera.py 불러온 것
import make_train_data as mtd
import pygame

#############################  fuctions definition #######################################
# 2. 빛을 지우고, face_landmark를 인식하기 쉽게 카메라 필터를 변환해줌.
# 3. 강도에 따라 다른 알람 파일 재생
# 4. 경로에서 alarm을 load 한 다음 재생.

# 2.
def light_removing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame을 gray로
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # 색반전
    L = lab[:, :, 0]  # frame 값을 리스트로 나열
    med_L = cv2.medianBlur(L, 99)  # frame의 노이즈 제거
    invert_L = cv2.bitwise_not(med_L)  # frame 빛반전
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)  # gray 75%, invert 25%로 new frame 생성
    return L, composed


# 3.
'''
label에 따라 알람이 다름.
0 power : 졸음 강도 강함 
1 normal : 졸음 강도 중간 
2 short : 졸음 강도 약함
'''


def select_alarm(result):
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1:
        sound_alarm("nomal_alarm.wav")
    else:
        sound_alarm("short_alarm.mp3")


# 4.
def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


class FaceRecog():  # 얼굴 인식을 위한 class
    def __init__(self):
        self.camera = camera.VideoCamera()  # camera.py의 VideoCamera 클래스

        self.image_face_encodings = []  # 사진의 얼굴 속성 값을 넣을 리스트
        self.image_face_names = []  # 사진의 얼굴 이름을 넣을 리스트
        self.is_recognized = 0  # 얼굴 인식이 완료되었는지 확인하는 변수(5 이상이 되면 얼굴 인식 완료)

        dirname = 'image'  # 얼굴 사진이 들어있는 디렉토리 이름
        files = os.listdir(dirname)  # listdir(): 디렉토리에 어떤 파일들이 있는지 리스트로 불러오기
        for filename in files:
            name, ext = os.path.splitext(filename)  # 파일이름을 2개의 이름으로 분리(이름, 확장자명)
            # 예를 들어, sein.jpg --> sein 과 .jpg
            if ext == '.jpg':  # 확장자가 jpg 면
                self.image_face_names.append(name)  # 얼굴 이름 리스트에 이름 추가
                pathname = os.path.join(dirname, filename)  # 파일이름을 경로에 합치기
                img = face_recognition.load_image_file(pathname)  # 위 경로를 통해 face_recognition에서 해당 이미지 불러오기
                face_encoding = face_recognition.face_encodings(img)[
                    0]  # 불러온 이미지에서 68개 얼굴 위치(face landmarks)의 속성 값 알아내기
                self.image_face_encodings.append(face_encoding)  # 얼굴 속성 값을 리스트에 저장

        self.face_locations = []  # 캠으로 인식한 얼굴 위치 값을 넣을 리스트
        self.face_encodings = []  # 캠으로 인식한 얼굴 속성 값을 넣을 리스트
        self.face_names = []  # 캠으로 인식한 얼굴 이름 값을 넣을 리스트
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):  # 순수 frame 가져오는 함수
        frame = self.camera.get_frame()
        return frame

    def get_face_frame(self):  # 얼굴 인식을 한 frame을 가져오는 함수

        frame = self.camera.get_frame()  # 캠으로부터 frame 읽어서
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # frame의 크기를 1/4로 줄임(계산량을 줄이기 위해)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR(OpenCV가 쓰는거) -> RGB(dlib의 face_recognition가 쓰는거)로 바꾸기

        if self.process_this_frame:  # 두 frame당 1번씩 계산(계산량을 줄이기 위해)

            self.face_locations = face_recognition.face_locations(rgb_small_frame)  # frame에서 얼굴 위치 추출
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame,
                                                                  self.face_locations)  # 얼굴 위치에서 face landmark 추출

            self.face_names = []
            for face_encoding in self.face_encodings:
                distances = face_recognition.face_distance(self.image_face_encodings,
                                                           face_encoding)  # 사진의 face landmark와 frame의 face landmark를 거리로 비교

                name = "Unknown"  # 거리가 0.6 이상이면 다른 사람으로 인식
                if distances < 0.6:  # 0.6 이하면
                    self.is_recognized += 1
                    index = np.argmin(distances)
                    name = self.image_face_names[index]  # 사진의 이름 불러오기
                    self.face_names.append(name)  # 이름 추가하기

        self.process_this_frame = not self.process_this_frame  # True면 False로, False면 True로

        # 결과 보여주는 네모 그리기(openCV 사용)
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # 얼굴 주위로 네모 박스 그리고

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  # 이름 붙이기
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def init_open_ear():
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target=init_message)
    th_message1.deamon = True
    th_message1.start()
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")


def init_close_ear():
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target=init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)  # EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :", EAR_THRESH, "\n")


def init_message():
    print("init_message")
    sound_alarm("init_sound.mp3")


#####################################################################################################################
# 1. Variables for checking EAR.
# 2. Variables for detecting if user is asleep.
# 3. When the alarm rings, measure the time eyes are being closed.
# 4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
# 5. We should count the time eyes are being opened for data labeling.
# 6. Variables for trained data generation and calculation fps.
# 7. Detect face & eyes.
# 8. Run the cam.
# 9. Threads to run the functions in which determine the EAR_THRESH.

# 1.
OPEN_EAR = 0  # For init_open_ear()
EAR_THRESH = 0  # Threashold value

# 2.
# It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 20
COUNTER = 0  # Frames counter.

# 3.
closed_eyes_time = []  # The time eyes were being offed.
TIMER_FLAG = False  # Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False  # Flag to check if alarm has ever been triggered.

# 4.
ALARM_COUNT = 0  # Number of times the total alarm rang.
RUNNING_TIME = 0  # Variable to prevent alarm going off continuously.

# 5.
PREV_TERM = 0  # Variable to measure the time eyes were being opened until the alarm rang.

# 6. make trained data
np.random.seed(9)
power, nomal, short = mtd.start(
    25)  # actually this three values aren't used now. (if you use this, you can do the plotting)
# The array the actual test data is placed.
test_data = []
# The array the actual labeld data of test data is placed.
result_data = []
# For calculate fps
prev_time = 0

# 7.
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 8.
# print("starting video stream thread...")
# vs = cv2.VideoCapture(0)
# time.sleep(1.0)


#####################################################################################################################
face_recog = FaceRecog()
print(face_recog.image_face_names)  # 사진의 이름 출력
is_first = True  # Thread는 한번만 실행되야 하기 때문에

while True:
    # is_recognized 가 5이상이면 얼굴이 인식되었다고 판단
    if face_recog.is_recognized < 5:
        frame = face_recog.get_face_frame()
        if face_recog.is_recognized == 5:
            print("your face is recognized!")

    else:
        # 9.
        if is_first == True:
            th_open = Thread(target=init_open_ear)
            th_open.deamon = True
            th_open.start()
            th_close = Thread(target=init_close_ear)
            th_close.deamon = True
            th_close.start()
            is_first = False

        frame = face_recog.get_frame()

        L, gray = light_removing(frame)

        rects = detector(gray, 0)

        # checking fps. If you want to check fps, just uncomment below two lines.
        # prev_time, fps = check_fps(prev_time)
        # cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # (leftEAR + rightEAR) / 2 => both_ear.
            both_ear = (leftEAR + rightEAR) * 500  # I multiplied by 1000 to enlarge the scope.

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if both_ear < EAR_THRESH:
                if not TIMER_FLAG:
                    start_closing = timeit.default_timer()
                    TIMER_FLAG = True
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:

                    mid_closing = timeit.default_timer()
                    closing_time = round((mid_closing - start_closing), 3)

                    if closing_time >= RUNNING_TIME:
                        if RUNNING_TIME == 0:
                            CUR_TERM = timeit.default_timer()
                            OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM), 3)
                            PREV_TERM = CUR_TERM
                            RUNNING_TIME = 1.75

                        RUNNING_TIME += 2
                        ALARM_FLAG = True
                        ALARM_COUNT += 1

                        print("{0}st ALARM".format(ALARM_COUNT))
                        print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                        print("closing time :", closing_time)
                        test_data.append([OPENED_EYES_TIME, round(closing_time * 10, 3)])
                        result = mtd.run([OPENED_EYES_TIME, closing_time * 10], power, nomal, short)
                        result_data.append(result)
                        t = Thread(target=select_alarm, args=(result,))
                        t.deamon = True
                        t.start()

            else:
                COUNTER = 0
                TIMER_FLAG = False
                RUNNING_TIME = 0

                if ALARM_FLAG:
                    end_closing = timeit.default_timer()
                    closed_eyes_time.append(round((end_closing - start_closing), 3))
                    print("The time eyes were being offed :", closed_eyes_time)

                ALARM_FLAG = False

            cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
# vs.stop()

