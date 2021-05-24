# coding: utf-8

# In[1]:


import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm


# 눈의 종횡비(EAR)를 구하는 함수. eye[0] = P1. eye[5] = P6.
# dist.euclidean: 두 점 사이의 유클리드 거리를 구해주는 함수

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# init_open_ear: 눈을 뜨고있을 때의 평균 EAR값 결정하는 함수.
def init_open_ear():
    time.sleep(5)  # sleep: 일시정지 함수.
    print("open init time sleep")
    ear_list = []  # ear_list: 측정된 EAR값이 저장될 리스트
    th_message1 = Thread(target=init_message)
    th_message1.deamon = True
    th_message1.start()
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR  # OPEN_EAR: 측정한 EAR의 평균 값이 저장될 변수
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")


# init_close_ear: 눈을 감고있을 때의 평균 EAR값 결정하는 함수.
def init_close_ear():
    time.sleep(2)  # sleep: 일시정지 함수
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []  # ear_list: 측정된 EAR값이 저장될 리스트
    th_message2 = Thread(target=init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)  # CLOSE_EAR: 측정한 EAR의 평균 값이 저장될 변수

    # EAR_THRESH: EAR의 50%.
    # if (EAR < EAR_THRESH): 운전자가 졸린 상태인 것으로 판단 -> 운전자가 수면 상태가 아니라도 알람 울림
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)  # EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :", EAR_THRESH, "\n")


# init_message: 알람 울리는 함수
def init_message():
    print("init_message")
    alarm.sound_alarm("init_sound.mp3")


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

# 2. 약 20프레임 이상 동안 EAR < EAR_THRESH 이면 운전자가 졸고 있다고 판단
# It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 20
COUNTER = 0  # Frames counter.

# 3. 알람이 울리면 눈을 감고있는 시간 측정.
closed_eyes_time = []  # The time eyes were being offed.
TIMER_FLAG = False  # Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False  # Flag to check if alarm has ever been triggered.

# 4. 알람이 울린 횟수 카운트
ALARM_COUNT = 0  # Number of times the total alarm rang.
RUNNING_TIME = 0  # Variable to prevent alarm going off continuously.

# 5. PREV_TERM: 알람이 울릴때까지 눈이 뜨는 시간을 측정하는 변수
PREV_TERM = 0  # Variable to measure the time eyes were being opened until the alarm rang.

# 6. make trained data
np.random.seed(9)
power, nomal, short = mtd.start(25)  # actually this three values aren't used now. (if you use this, you can do the plotting)
# The array the actual test data is placed.
test_data = []
# The array the actual labeld data of test data is placed.
result_data = []
# For calculate fps
prev_time = 0

# 7. 랜드마크를 이용해 얼굴&눈 탐지
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 8. 캠 켜기
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# 9. EAR_THRESH를 결정하기 위한 함수를 실행시키는 thread
th_open = Thread(target=init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target=init_close_ear)
th_close.deamon = True
th_close.start()

#####################################################################################################################

while True:
    frame = vs.read() #캠에서 받은 frame
    frame = imutils.resize(frame, width=400) #frame의 크기 조정

    L, gray = lr.light_removing(frame) #light_remover를 이용해 frame에서 조명 제거

    rects = detector(gray, 0) # frame에서 grayscale frame 탐지

    # checking fps. If you want to check fps, just uncomment below two lines.
    # prev_time, fps = check_fps(prev_time)
    # cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd] # 왼쪽 눈
        rightEye = shape[rStart:rEnd] # 오른쪽 눈
        leftEAR = eye_aspect_ratio(leftEye) # 왼쪽 눈의 ear
        rightEAR = eye_aspect_ratio(rightEye) # 오른쪽 눈의 ear

        # (leftEAR + rightEAR) / 2 => both_ear.
        both_ear = (leftEAR + rightEAR) * 500  # I multiplied by 1000 to enlarge the scope. # 두 눈의 평균 ear

        leftEyeHull = cv2.convexHull(leftEye) # 점으로 다각형 구하기
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # frame에 눈 위치 표시
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if both_ear < EAR_THRESH: # 두눈의 평균 ear이 EAR_THRESH보다 작으면, 즉 눈을 감고있으면
            if not TIMER_FLAG: # 지금까지 한번도 졸지 않았다면
                start_closing = timeit.default_timer() # 눈 감고 있는 시간 측정 시작
                TIMER_FLAG = True
            COUNTER += 1 # 눈 감고있는 frame이 20이상이면 졸고있다고 측정하기 위해 counter 증가

            if COUNTER >= EAR_CONSEC_FRAMES: # 눈 감고있는 frame이 20이상이 되면

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

                    print("{0}st ALARM".format(ALARM_COUNT)) # 몇번째 알람입니다
                    print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME) # 알람이 울리기 전에 눈이 떠진 시간
                    print("closing time :", closing_time)
                    test_data.append([OPENED_EYES_TIME, round(closing_time * 10, 3)])
                    result = mtd.run([OPENED_EYES_TIME, closing_time * 10], power, nomal, short)
                    result_data.append(result)
                    t = Thread(target=alarm.select_alarm, args=(result,))
                    t.deamon = True
                    t.start()

        else: # 눈을 뜨고 있으면
            COUNTER = 0 # 초기화
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG: # 전에 알람이 울린 적이 있으면
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing - start_closing), 3))
                print("The time eyes were being offed :", closed_eyes_time)

            ALARM_FLAG = False

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 30, 20), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"): # q누르면 종료
        break

cv2.destroyAllWindows()
vs.stop()

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    L, gray = lr.light_removing(frame)

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
                    t = Thread(target=alarm.select_alarm, args=(result,))
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

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 30, 20), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()