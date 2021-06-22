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
# 1. 강도에 따라 다른 알람 파일 재생
# 2. 경로에서 alarm을 load 한 다음 재생.


# 1.
'''
label에 따라 알람이 다름.
0 weak : 졸음 강도 약함 
1 strong : 졸음 강도 강함 
'''
def def_alarm(result):
    if result == 0:
        play_sound("nomal_alarm.wav")
    elif result == 1:
        play_sound("ppi.mp3")


# 2.
def play_sound(path):
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


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = L / D
    return mar


def init_open_ear():
    time.sleep(5)
    print("눈을 떠주세요")
    ear_list = []
    play_sound("open_your_eyes.mp3")
    th_ring1 = Thread(target=play_sound("ppi.mp3"))
    th_ring1.start()
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
    print("눈을 감아주세요")
    ear_list = []
    play_sound("close_your_eyes.mp3")
    th_ring2 = Thread(target=play_sound("ppi.mp3"))
    th_ring2.start()
    time.sleep(1)
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)  # EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :", EAR_THRESH, "\n")


def init_open_mouth():
    time.sleep(2)
    th_close.join()
    time.sleep(5)
    print("입을 벌려주세요")
    mar_list = []
    play_sound("open_your_mouth.mp3")
    th_ring3 = Thread(target=play_sound("ppi.mp3"))
    th_ring3.start()
    for i in range(7):
        mar_list.append(mouth_mar)
        time.sleep(1)
    global OPEN_MAR
    OPEN_MAR = sum(mar_list) / len(mar_list)
    print("open mouth =", mar_list, "\nOPEN_MAR =", OPEN_MAR, "\n")


def init_close_mouth():
    time.sleep(2)
    mouth_open.join()
    time.sleep(5)
    print("입을 다물어주세요")
    mar_list = []
    play_sound("close_your_mouth.mp3")
    th_ring4 = Thread(target=play_sound("ppi.mp3"))
    th_ring4.start()
    time.sleep(1)
    for i in range(7):
        mar_list.append(mouth_mar)
        time.sleep(1)
    CLOSE_MAR = sum(mar_list) / len(mar_list)
    global MAR_THRESH
    MAR_THRESH = ((OPEN_MAR - CLOSE_MAR) * 0.7) + CLOSE_MAR
    print("close mouth =", mar_list, "\nCLOSE_MAR =", CLOSE_MAR, "\n")
    print("The last MAR_THRESH's value : ", MAR_THRESH, "\n")

''' 
졸음 강도를 결정하는 함수
c_time: 눈 감은 시간
result: 졸음 강도 (0,1)
눈 감은 시간이 2초 이상일 경우, lv0 (weak alarm)
눈 감은 시간이 4초 이상일 경우, lv1 (strong alarm)
'''
def def_level(c_time):
    result = -1
    if c_time >= 2:
        result = 0
        if c_time >= 4:
            result = 1
    print("drowsiness level :", result)
    return result


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
EAR_THRESH = 0  # EAR 기준값
MAR_THRESH = 0
OPEN_MAR = 0

# 2.
# It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 20
COUNTER = 0  # 졸음 프레임 카운터
YAWN_COUNT = -1  # 하품 횟수 카운트

# 3.
closed_eyes_time = []  # 눈을 감은 시간
TIMER_FLAG = False  # Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False  # Flag to check if alarm has ever been triggered.
YAWN_FLAG = False
YAWN_TIMER = False
# 4.
ALARM_COUNT = 0  # Number of times the total alarm rang.
# RUNNING_TIME = 0  # Variable to prevent alarm going off continuously.

# 5.
# PREV_TERM = 0  # Variable to measure the time eyes were being opened until the alarm rang.

# 6. make trained data
np.random.seed(9)
# The array the actual test data is placed.
test_data = []
# The array the actual labeld data of test data is placed.
result_data = []
# For calculate fps
# prev_time = 0

# 7. 랜드마크 추출
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# 8.
# print("starting video stream thread...")
# vs = cv2.VideoCapture(0)
# time.sleep(1.0)


#####################################################################################################################
face_recog = FaceRecog()
print(face_recog.image_face_names)  # 사진의 이름 출력
is_first = 0  # Thread는 한번만 실행되야 하기 때문에

while True:
    # is_recognized 가 5이상이면 얼굴이 인식되었다고 판단
    if face_recog.is_recognized < 5:
        frame = face_recog.get_face_frame()
        if face_recog.is_recognized == 5:
            print("your face is recognized!")

    else:
        # 9.
        if is_first == 0:  # 처음 실행되는 경우, EAR 값 초기화
            th_open = Thread(target=init_open_ear)
            th_open.start()
            th_close = Thread(target=init_close_ear)
            th_close.start()
            mouth_open = Thread(target=init_open_mouth)
            mouth_open.start()
            mouth_close = Thread(target=init_close_mouth)
            mouth_close.start()

            is_first = 1

        frame = face_recog.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 영상 회색조 처리

        rects = detector(gray, 0)

        # checking fps. If you want to check fps, just uncomment below two lines.
        # prev_time, fps = check_fps(prev_time)
        # cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

        # 영상에서 랜드마크를 추출하여 눈의 위치 파악
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mar = mouth_aspect_ratio(mouth)

            # (leftEAR + rightEAR) / 2 => both_ear.
            both_ear = (leftEAR + rightEAR) / 2.0  # I multiplied by 1000 to enlarge the scope.
            mouth_mar = mar

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # EAR이 역치보다 작으면
            if both_ear < EAR_THRESH:
                if not TIMER_FLAG:  # 만약 타이머가 작동되지 않는 상태면 졸음시간 측정 시작
                    start_closing = timeit.default_timer()
                    TIMER_FLAG = True
                COUNTER += 1  # 프레임 카운트 시작

                # 일정 시간 이상 눈을 감고 있으면
                if COUNTER >= EAR_CONSEC_FRAMES:
                    mid_closing = timeit.default_timer()
                    closing_time = round((mid_closing - start_closing), 3)
                    level = def_level(closing_time)
                    def_alarm(level)  # 알람 울림
                    time.sleep(5)  # 알람이 울릴동안 일시정지
                    # if (result==0) result += 1

                    # if closing_time >= RUNNING_TIME:
                    # if RUNNING_TIME == 0:
                    # CUR_TERM = timeit.default_timer()
                    # OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM), 3)
                    # PREV_TERM = CUR_TERM
                    # RUNNING_TIME = 1.75

                    # RUNNING_TIME += 2
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

            #                    result = def_level(closing_time)  # 알람 레벨값 저장
            #                    result_data.append(result)
            #                    def_alarm(result)

            # print("{0}st ALARM".format(ALARM_COUNT))
            # print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
            # print("closing time :", closing_time)
            # test_data.append([OPENED_EYES_TIME, round(closing_time * 10, 3)])
            # result = mtd.run([OPENED_EYES_TIME, closing_time * 10], power, nomal, short)
            # result_data.append(result)

            else:
                COUNTER = 0
                TIMER_FLAG = False
                # RUNNING_TIME = 0

                if ALARM_FLAG:
                    end_closing = timeit.default_timer()  # 졸음이 끝난 시간 측정
                    closed_eyes_time.append(round((end_closing - start_closing), 3))  # 졸음 시간 계산
                    print("The time eyes were being offed :", closed_eyes_time)

                ALARM_FLAG = False

            cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

            if mar > MAR_THRESH:
                if not YAWN_TIMER:
                    start_yawn = timeit.default_timer()
                    YAWN_TIMER = True

                if (timeit.default_timer() - start_yawn > 2.0):
                    if not YAWN_FLAG:
                        YAWN_COUNT += 1
                        YAWN_FLAG = True



            else:
                YAWN_TIMER = False
                YAWN_FLAG = False

            cv2.putText(frame, "MAR : {:.2f}".format(mar), (300, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)
            text = "YAWN COUNT : " + str(YAWN_COUNT)
            cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 30, 20), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
# vs.stop()
