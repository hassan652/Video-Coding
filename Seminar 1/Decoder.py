import numpy as np
import cv2
import pickle 
import time
		
#Color conversion(inverse)
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    x=rgb.dot(xform.T)
    return x

try:
  f1 = open ('video_raw_data.txt', 'rb')
  f2 = open ('video_raw_data_int8.txt', 'rb')
  f3 = open ('video_raw_data_unint8.txt', 'rb')

except IOError:
    print('No such file')
    
while(True):
    reduced = pickle.load (f1)
    frame =reduced.copy()
    converted = ycbcr2rgb(frame)
   # cv2.imshow ('Y-component', frame/255)
    cv2.imshow ('Converted Video', converted/255)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
