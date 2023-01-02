import random
import cv2

import numpy as np

import glob
import re

import os

QUIZ_NUM = 30

def adjust(img, alpha=1.0, beta=0.0):
  # 積和演算を行う。
  dst = alpha * img + beta
  # [0, 255] でクリップし、uint8 型にする。
  return np.clip(dst, 0, 255).astype(np.uint8)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_black():
  size = (320,240)
  return np.zeros(size,np.uint8)

def show_pic(files_a,files_b,a_num,b_num):

  max_length = max(len(files_a),len(files_b))

  for repeat in range(3):
    for t in range(1,max_length):
      if(len(files_a)>t):
        filename_a = "/home/kazumi/download/face_test/picture/"+str(a_num)+"_"+str(t)+".jpg"
      else:
        filename_a = "/home/kazumi/download/face_test/picture/"+str(a_num)+"_"+str(len(files_a))+".jpg"
      if(len(files_b)>t):
        filename_b = "/home/kazumi/download/face_test/picture/"+str(b_num)+"_"+str(t)+".jpg"
      else:
        filename_b = "/home/kazumi/download/face_test/picture/"+str(b_num)+"_"+str(len(files_b))+".jpg"

      img_a=cv2.imread(filename_a)#,cv2.IMREAD_GRAYSCALE)
      img_b=cv2.imread(filename_b)#,cv2.IMREAD_GRAYSCALE)
      img = cv2.hconcat([img_a,img_b])
      cv2.imshow('window',img)
      cv2.waitKey(30)
  cv2.destroyAllWindows()

def play_video(a,b):
  PIC_PATH_A = "/home/kazumi/download/face_test/picture/"+str(a)+"_*.jpg"
  files_a = sorted(glob.glob(PIC_PATH_A), key=natural_keys)
  PIC_PATH_B = "/home/kazumi/download/face_test/picture/"+str(b)+"_*.jpg"
  files_b = sorted(glob.glob(PIC_PATH_B), key=natural_keys)
  show_pic(files_a,files_b,a,b)

  print("number",a,b,";input larger number: a or b or =")
  high_number = input()
  a_val = 0
  b_val = 0
  if(high_number == 'a'):
    a_val = 10
    b_val = 0
  elif(high_number == 'b'):
    a_val = 0
    b_val = 10
  elif(high_number == '='):
    a_val = 5
    b_val = 5
  return a_val,b_val
 
# クイックソートを行う関数
def quick_sort(x):
    # 基準値を抽出(半分の位置の値)
    n = len(x)
    pivot = x[int(n / 2)]
 
    # i番目の値と基準値を比較して左l、右r、真ん中mに追加
    l = []
    r = []
    m = []
    print(x)
    for i in range(n):
        sample = x[i]
        sample_val,pivot_val=play_video(sample,pivot)
        if sample_val < pivot_val:
        #if sample< pivot:
            l.append(sample)
        elif sample_val > pivot_val:
        #elif sample> pivot:
            r.append(sample)
        else:
            m.append(sample)
    # lとrの場合でそれぞれ再帰処理による分割を行う
    if l:
        l = quick_sort(l)
    if r:
        r = quick_sort(r)
    return l + m + r
 
# データをランダムに用意してソートを実行
#x = random.sample(range(100), k=QUIZ_NUM)
x = range(QUIZ_NUM)
list = quick_sort(x)
print('initial array:', x)
print('quick sorted:', list)
