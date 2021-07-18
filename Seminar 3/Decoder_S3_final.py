import pickle
import numpy as np
import scipy.signal as sp
import cv2
import scipy.fftpack as sft

def inverse_dct(frame):

    r,c= frame.shape
    X=np.reshape(frame,(-1,8), order='C')
    X=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T

    return x

def upSample(downSampled,N):
    r, c = downSampled.shape
    sampleFrame = np.zeros((r * N, c * N))
    sampleFrame[0::N,0::N] = downSampled
    return sampleFrame

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb =rgb.dot(xform.T)
    #return np.uint8(rgb)
    return rgb

def ycbcr_to_rgb(frame):
    RGB = np.matrix([[0.299, 0.587, 0.114],
                     [0.16864, -0.33107, 0.49970],
                     [0.499813, -0.418531, -0.081282]])

    YCbCr = RGB.I
    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr, frame[i].T).T#/255.

    return xframe
'''
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = frame.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    # return np.uint8(rgb)
    return rgb
'''
'''
    RGB = np.matrix([[0.299, 0.587, 0.114],
                     [-0.16864, -0.33107, 0.49970],
                     [0.499813, -0.418531, -0.081282]])

    YCbCr = RGB.I

    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr, frame[i].T).T#/255.

    return xframe
    '''
N = 2

#Lowpass filter
rect_filt = np.ones((N, N)) / N
pyra_filt = sp.convolve2d(rect_filt, rect_filt) / N

f_hand = open('DCTWithoutZeros.txt', 'rb')

try:
    while True:
       #Loading Y, Cb and Cr components from the file
       DCTY = pickle.load(f_hand)
       DCTCb = pickle.load(f_hand)
       DCTCr = pickle.load(f_hand)
       #frame=pickle.load(f_hand)

       #Applying inverse DCT using the defined function
       InverseY = inverse_dct(DCTY)
       InverseCb = inverse_dct(DCTCb)
       InverseCr = inverse_dct(DCTCr)

       #Upsampling the components
       Upsampled_Cb = upSample(InverseCb, N)
       Upsampled_Cr = upSample(InverseCr, N)

       #Dec components
       Cb_dec = sp.convolve2d(Upsampled_Cb, pyra_filt, mode="same")
       #print("shape of CB dec", Cb_dec.shape)
       Cr_dec = sp.convolve2d(Upsampled_Cr, pyra_filt, mode="same")

       #Combining the Dec components
       rows, cols = DCTY.shape
       Dec_YCbCr = np.zeros((rows, cols, 3))
       #print("shape of Dec frame", Dec_YCbCr.shape)
       Dec_YCbCr[:, :, 0] = InverseY
       Dec_YCbCr[:, :, 1] = Cb_dec
       Dec_YCbCr[:, :, 2] = Cr_dec

       #r = (InverseY+Cr_dec*1.4025)
       #g = (InverseY+Cb_dec*(-0.34434)+Cr_dec*(-0.7144))
       #b = (InverseY+Cb_dec*1.7731)

       #frameRecover = r,c,3
       #frameRecover = np.zeros((r, c, 3))
       #frameRecover[:,:,0],frameRecover[:,:,1],frameRecover[:,:,2] = b,g,r
       #cv2.imshow('RGB Recover',frameRecover/255)

       #Dec components to RGB frame using the defined function
       RGB= ycbcr_to_rgb(Dec_YCbCr.astype(float))

       #Displaying reconstructed RGB frame
       cv2.imshow("Reconstructed RGB",RGB)
       if cv2.waitKey(200) & 0xFF == ord('q'):
           break


except EOFError:
    pass
