import numpy as np
import cv2
import pickle
import time
import os

#Conversion from RGB to YCbCr
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.16864, -.33107, .49970], [.499813, -.418531, -.081282]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    #return np.uint8(ycbcr)
    return ycbcr

#Conversion from YCbCr to RGB 
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb =rgb.dot(xform.T)
    #return np.uint8(rgb)
    return rgb

cap = cv2.VideoCapture(0)

try:
  f1=open('video_raw_data.txt', 'wb')
  f2=open('video_raw_data_Cb-Cr_int8.txt', 'wb')
  f3=open('video_raw_data_Y_unint8.txt', 'wb')
  f4=open('A_video_raw_data_int8.txt', 'wb')
  f5=open('B_video_raw_data_uint8.txt', 'wb')
except IOError:
   print('No such file')
#storing the video for 3 seconds only

#t_end = time.time() + 3
#while time.time() < t_end:
while(True):

    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), None, .8, .8)
    converted = rgb2ycbcr(frame)
    converted = cv2.resize(converted, (0, 0), None, .65, .65)
    cv2.imshow ('Original Video', frame / 512)
    cv2.imshow ('Y-component', converted [:, :, 0] / 255)
    cv2.imshow ('Cb', converted [:, :, 1] / 255)
    cv2.imshow ('Cr', converted [:, :, 2] / 255)
    resized = ycbcr2rgb (converted)
    resized = cv2.resize(resized, (0, 0), None, 1.53846, 1.53846)
    cv2.imshow ('Resized Video', resized / 255)


    pickle.dump(converted, f1)
#'''
        #pickle dumping as int8

    reduced3 = np.array(converted, dtype='int8')
    pickle.dump(reduced3, f4)

    # pickle dumping as unit8

    reduced4 = np.array(converted, dtype='uint8')
    pickle.dump(reduced4, f5)

# '''
        # pickle dumping Cb and Cr components
    reduced1 = np.array (converted[:,:,[1,2]], dtype='int8')
    pickle.dump (reduced1, f2)

        #pickle dumping Y component
    reduced2 = np.array (converted[:, :, 0], dtype='uint8')
    pickle.dump (reduced2, f3)

    if (cv2.waitKey (30) & 0xFF == ord ('q')):
        break
#file sizes in Mbs
raw = ((os.path.getsize("video_raw_data.txt")))
raw_int8 = (os.path.getsize("video_raw_data_Cb-Cr_int8.txt"))
raw_uint8 = (os.path.getsize("B_video_raw_data_uint8.txt"))
#print(dtype('video_raw_data.txt'))
print('Original File Size = ' , int(raw)/1000000 , 'Mb ---',
      'Int8 File size =' , int(raw_int8)/1000000, 'Mb ---') #, 'Raw data file uint8 = ' ,raw_uint8, 'bytes ---')



cap.release()
f1.close()
f2.close()
f3.close()
cv2.destroyAllWindows()
