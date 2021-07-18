import cv2
import numpy as np
import scipy.signal as sp
import pickle
import scipy.fftpack as sft


#Functions
def RGB2YCBCR(frame):
    R = frame[:, :, 2]
    G = frame[:, :, 1]
    B = frame[:, :, 0]
    Y = (0.299 * R + 0.587 * G + 0.114 * B)
    Cb = (-0.16864 * R - 0.33107 * G + 0.49970 * B)
    Cr = (0.499813 * R - 0.418531 * G - 0.081282 * B)
    return Y, Cb, Cr

def chromaSubSampleWithoutZero(Cb, Cr):
    subCb = Cb[0::2, 0::2]
    subCr = Cr[0::2, 0::2]
    return subCb, subCr

DTCFactor = 4

def dct(frame):
    r, c = frame.shape
    Mr = np.concatenate([np.ones(DTCFactor), np.zeros(8 - DTCFactor)])
    # print "Mr: \n", Mr
    # Mr[int(8/4.0):r]=np.zeros(int(3.0/4.0*8))

    Mc = Mr
    # frame=np.reshape(frame[:,:,1],(-1,8), order='C')
    frame = np.reshape(frame, (-1, 8), order='C')
    X = sft.dct(frame / 255.0, axis=1, norm='ortho')
    # apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X = np.dot(X, np.diag(Mr))
    # shape it back to original shape:
    X = np.reshape(X, (-1, c), order='C')
    # Shape frame with columns of hight 8 by using transposition .T:
    X = np.reshape(X.T, (-1, 8), order='C')
    X = sft.dct(X, axis=1, norm='ortho')
    # apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X = np.dot(X, np.diag(Mc))
    # shape it back to original shape:
    X = (np.reshape(X, (-1, r), order='C')).T
    # Set to zero the 7/8 highest spacial frequencies in each direction:
    # X=X*M
    return X


originalFile= open('original.txt','wb')
#DCTWithZerosFile=open('DCTWithZeros.txt','wb')
DCTWithoutZerosFile= open('DCTWithoutZeros.txt','wb')

N=2
#Lowpass filter
rect_filt = np.ones((N, N)) / N
pyra_filt = sp.convolve2d(rect_filt, rect_filt) / N
cap = cv2.VideoCapture(0)
frame=0  #Why

while (True):
    ret, frame = cap.read()

    cv2.imshow("original", frame)
    frame = frame.astype(np.float32)

    Y, Cb, Cr = RGB2YCBCR(frame)
    YCbCr = np.zeros(frame.shape)
    #Assigning Y, Cb and Cr values
    YCbCr[:, :, 0] = Y
    YCbCr[:, :, 1] = Cb
    YCbCr[:, :, 2] = Cr


    pickle.dump(YCbCr, originalFile)
    # cv2.imshow("cr",Cr)
    # cv2.imshow("Y_component",Y)
    pyfCb = sp.convolve2d(Cb, pyra_filt, mode='same')
    pyfCr = sp.convolve2d(Cr, pyra_filt, mode='same')
    #cv2.imshow("low_passed_cb",pyfCb)

    #Downsampling Cb and Cr components and showing them
    Cb_Ds, Cr_Ds = chromaSubSampleWithoutZero(pyfCb, pyfCr)
    cv2.imshow("Downsampled Cb", Cb_Ds)
    cv2.imshow("Downsampled Cr", Cr_Ds)

    dct_y = dct(Y)
    # dct_y = np.float32(dct_y)/255.0
    dct_cb = dct(Cb_Ds)
    # dct_cb = np.float32(dct_cb)/255.0
    dct_cr = dct(Cr_Ds)
    # dct_cr=np.float32(dct_cr)/255.0
    # cv2.imshow("dct_cb",np.abs(dct_cb))
    # cv2.imshow("dct_cr",np.abs(dct_cr))
    ########
    pickle.dump(dct_y, DCTWithoutZerosFile)
    pickle.dump(dct_cb, DCTWithoutZerosFile)
    pickle.dump(dct_cr, DCTWithoutZerosFile)

    frame += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

compressed = DCTWithoutZerosFile.tell() / 1024.0 ** 2
original = originalFile.tell() / 1024.0 ** 2

print('File size of the DCT Compressed Video data')
print(compressed, 'Mb')

print('File size of the actual Video data')
print(original, 'Mb')
print('Compression factor')
print(original / compressed)

cap.release()
cv2.destroyAllWindows()
DCTWithoutZerosFile.close()
DCTWithoutZerosFile.close()

