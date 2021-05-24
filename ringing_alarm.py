# coding: utf-8

# In[ ]:


import pygame

'''
label에 따라 알람이 다름.
0 power : 졸음 강도 강함 
1 normal : 졸음 강도 중간 
2 short : 졸음 강도 약함

sound_alarm 함수를 통해 파일 재생
'''
def select_alarm(result):
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1:
        sound_alarm("nomal_alarm.wav")
    else:
        sound_alarm("short_alarm.mp3")

# 경로 load 한 다음 재생.
def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


