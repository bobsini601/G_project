
# coding: utf-8

# In[ ]:


import time

def check_fps(prev_time) :
    cur_time = time.time() #Import the current time in seconds. 현재 시간 체크.
    one_loop_time = cur_time - prev_time #주기 = 현재시간 - 기준 시간
    prev_time = cur_time #기준 시간 갱신
    fps = 1/one_loop_time #초당 프레임 수 fps
    return prev_time, fps
