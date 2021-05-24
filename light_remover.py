import cv2

def light_removing(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame을 gray로
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # 검색해보니까 색반전? 느낌인듯
    L = lab[:,:,0] # frame 값을 리스트로 나열
    med_L = cv2.medianBlur(L,99) #median filter # frame의 노이즈 제거
    invert_L = cv2.bitwise_not(med_L) #invert lightness # frame 빛반전
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0) #gray 75%, invert 25%로 new frame 생성
    return L, composed