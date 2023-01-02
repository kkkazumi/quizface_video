import cv2
import numpy as np

import glob
import re

def adjust(img, alpha=1.0, beta=0.0):
  # 積和演算を行う。
  dst = alpha * img + beta
  # [0, 255] でクリップし、uint8 型にする。
  return np.clip(dst, 0, 255).astype(np.uint8)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def show_pic(files_a,a_num):

  max_length = len(files_a)

  print('quiznum',a_num)
  for t in range(1,max_length):
    filename_a = "/home/kazumi/download/face_test/picture/"+str(a_num)+"_"+str(t)+".jpg"

    img_a=cv2.imread(filename_a)#,cv2.IMREAD_GRAYSCALE)
    cv2.imshow('window',img_a)
    cv2.waitKey(30)
  cv2.destroyAllWindows()
  input()

def play_video(a):
  PIC_PATH_A = "/home/kazumi/download/face_test/picture/"+str(a)+"_*.jpg"
  files_a = sorted(glob.glob(PIC_PATH_A), key=natural_keys)
  show_pic(files_a,a)

order = np.loadtxt('sorted_order.txt')
for i in order:
  print(int(i))
  play_video(int(i))
