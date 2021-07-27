import cv2
import pickle
import numpy as np
import scipy.signal as sp
import scipy.fftpack as sft
import scipy.signal

#Functions
def ycbcr2rgb(frame):
    Y = frame[:, :, 0]
    Cb = frame[:, :, 1] - 127
    Cr = frame[:, :, 2] - 127

    R = 1 * Y + 0 * Cb + 1.4025 * Cr
    G = 1 * Y - 0.34434 * Cb - 0.7144 * Cr
    B = 1 * Y + 1.7731 * Cb + 0 * Cr

    rgb = np.zeros(frame.shape)
    rgb[:, :, 0] = B
    rgb[:, :, 1] = G
    rgb[:, :, 2] = R

    return rgb

def lowpass_filter(frame):
    # low pass filtering done with a cut off frequency of 1/Nth of the orignal nyquist frequency
    # filter kernel
    filt1 = np.ones((2, 2)) / 2
    filt2 = scipy.signal.convolve2d(filt1, filt1) / 2

    f_cb = scipy.signal.convolve2d(frame[:, :, 1], filt2, mode='same')  # Applying Filter to Cb component
    f_cr = scipy.signal.convolve2d(frame[:, :, 2], filt2, mode='same')  # Applying Filter to Cr component

    filtered_frame = np.zeros(frame.shape)
    filtered_frame[:, :, 0] = frame[:, :, 0]
    filtered_frame[:, :, 1] = f_cb
    filtered_frame[:, :, 2] = f_cr

    return filtered_frame

def inverse_dct(X):
    r , c = X.shape
    #Rows:
    X=np.reshape(X,(-1,8), order='C')
    X=sft.idct(X*255,axis=1,norm='ortho')  #*255
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    #Columns
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T
    return x

def dct_uncompress_1by4(zr_dctcomponent):
    row, col = zr_dctcomponent.shape

    # Keeping 4 values from 8x8 DCT block
    recons = np.zeros((int(row * 4), int(col * 4)))
    recons[::8, ::8] = zr_dctcomponent[::2, ::2]
    recons[1::8, ::8] = zr_dctcomponent[1::2, ::2]
    recons[0::8, 1::8] = zr_dctcomponent[0::2, 1::2]
    recons[1::8, 1::8] = zr_dctcomponent[1::2, 1::2]

    return recons


f = open('compressed.txt', 'rb')
num = 0
try:
    while True:
        for i in range(3):
            read = pickle.load(f)

            if i == 0:
                Y = read
                Y = dct_uncompress_1by4(Y)
                Y = inverse_dct(Y)

                r, c = Y.shape
                upsampled = np.zeros(shape=(r, c, 3))
                upsampled[:, :, 0] = Y
                print("decoding y")

            elif i == 1:
                print("Decoding Cb")
                cb = read
                cb = dct_uncompress_1by4(cb)
                cb = inverse_dct(cb)
                #Entering zeros, Upsampling
                upsampled[0::2, 0::2, 1] = cb


            elif i == 2:
                print("Decoding Cr")
                cr = read
                cr = dct_uncompress_1by4(cr)
                cr = inverse_dct(cr)
                upsampled[0::2, 0::2, 2] = cr  # upsampled, zeros entered

        num += 1
        print("Reconstructing the frame number: ", num)
        # filtering after upsampling
        recons = lowpass_filter(upsampled)

        # conversion to rgb
        rgb = ycbcr2rgb(recons)
        cv2.putText(rgb, f"Frame I= {num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
        cv2.imshow('Reconstructed I Frame', rgb / 255)

        ##############################################
        ######## DECODING MOTION VECTORS #############
        ##############################################
        print("decoding MV")
        #Getting Motion Vectors
        mv = pickle.load(f)
        add_P_frame = recons.copy()  # old frame copied

        for i in range(0, mv.shape[0]):
            for j in range(0, mv.shape[1]):
                if mv[i, j, 0] != 0 or mv[i, j, 1] != 0:  # means motion detected
                    old_pos = np.array([i * 8 + 8, j * 8 + 8])
                    new_pos = mv[i, j, :].astype(np.int) + old_pos
                    add_P_frame[old_pos[0] - 8: old_pos[0] + 8, old_pos[1] - 8: old_pos[1] + 8, :] = recons[
                                                                                                     new_pos[0] - 8:
                                                                                                     new_pos[0] + 8,
                                                                                                     new_pos[1] - 8:
                                                                                                     new_pos[1] + 8, :]

        num += 1
        print("Constructing Frame no. ", num)
        rgbp = ycbcr2rgb(add_P_frame)
        cv2.putText(rgbp, f"Frame P= {num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Reconstructed P Frame', rgbp / 255)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
except:
    print("Ending Program: Ran out of input")
cv2.destroyAllWindows()
f.close()