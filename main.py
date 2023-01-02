import os
import cv2
import numpy as np

import glob
import re



img_outdir = './video'
os.makedirs(img_outdir, exist_ok=True)

def adjust(img, alpha=1.0, beta=0.0):
  # 積和演算を行う。
  dst = alpha * img + beta
  # [0, 255] でクリップし、uint8 型にする。
  return np.clip(dst, 0, 255).astype(np.uint8)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

for quiz_num in range(30):
  videoname=img_outdir+"/"+"video_"+str(quiz_num)+".mp4"

  PIC_PATH = "/home/kazumi/download/face_test/picture/"+str(quiz_num)+"_*.jpg"
  files = sorted(glob.glob(PIC_PATH), key=natural_keys)


  # 動画作成
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  fps=30.0
  video  = cv2.VideoWriter(videoname, fourcc, fps, (320,240))

  #outimg_files = []
  for filename in files:
    #print('filename',filename)
    img = cv2.imread(filename)
    #outimg_files.append(filename)
    video.write(img)

  #height, width, channels = img.shape[:3]
  #print(height,width)
  #for img_file in outimg_files:
  #    img = cv2.imread(img_file)
  #    #cv2.imshow('frame',img)
  #    #cv2.waitKey(1)
  video.release()
  print(videoname)
