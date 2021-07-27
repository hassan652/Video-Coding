import scipy.signal
import scipy.fftpack as sft
import os
import numpy as np
import cv2
import scipy.signal as sp
import pickle

#Functions
def rgb2ycbcr(frame):
    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]
    y = 0.299*r + 0.587*g + 0.114*b
    cb = (-0.16864*r - 0.33107*g + 0.49970*b) +127
    cr = (0.499813*r - 0.418531*g - 0.081282*b) +127
    ycbcr = np.zeros(frame.shape)
    ycbcr[:,:,0] = y
    ycbcr[:,:,1] = cb
    ycbcr[:,:,2] = cr
    return ycbcr

def downsampling(frame):  # in place of frame ycbcr frame is entered
    # downsampling
    r, c, d = frame.shape

    y = frame[:, :, 0]  # y component remains same

    dscb = np.zeros((int(r / 2), int(c / 2)))
    dscb = frame[::2, ::2, 1]  # downsampling Cb and removing zeros

    dscr = np.zeros((int(r / 2), int(c / 2)))
    dscr = frame[::2, ::2, 2]  # downsampling Cr and removing zeros

    return y, dscb, dscr

def motionvector(Y, Yprev):
    rows, columns = Y.shape
    framevectors = np.zeros((rows, columns, 3))

    mv = np.zeros((int(rows / 8), int(columns / 8), 2))

    block = np.array([8, 8])
    for yblock in range(10):
        block[0] = yblock * 8 + 8;
        for xblock in range(10):
            block[1] = xblock * 8 + 8;
            # current block:
            Yc = Y[block[0]:block[0] + 8, block[1]:block[1] + 8]

            bestmae = 1000

            for ymv in range(-8, 8):
                for xmv in range(-8, 8):
                    diff = Yc - Yprev[block[0] + ymv: block[0] + ymv + 8, block[1] + xmv: block[1] + xmv + 8]
                    mae = sum(sum(np.abs(diff))) / 64
                    if mae <= bestmae:
                        bestmae = mae;
                        mv[yblock, xblock, 0] = ymv
                        mv[yblock, xblock, 1] = xmv
    return mv

def lowpass_filter(frame):
    # filter kernel
    # low pass filtering done with a cut off frequency of 1/Nth of the original nyquist frequency
    filt1 = np.ones((2, 2)) / 2
    filt2 = scipy.signal.convolve2d(filt1, filt1) / 2

    f_cb = scipy.signal.convolve2d(frame[:, :, 1], filt2, mode='same')  # Applying Filter to Cb component
    f_cr = scipy.signal.convolve2d(frame[:, :, 2], filt2, mode='same')  # Applying Filter to Cr component

    filtered_frame = np.zeros(frame.shape)
    filtered_frame[:, :, 0] = frame[:, :, 0]
    filtered_frame[:, :, 1] = f_cb
    filtered_frame[:, :, 2] = f_cr

    return filtered_frame

def dct_compress_1by4(dctcomponent):
    r, c = dctcomponent.shape

    # Keeping 4 values from 8x8 DCT block
    dct_zr = np.zeros((int(r / 4), int(c / 4)))
    dct_zr[::2, ::2] = dctcomponent[::8, ::8]
    dct_zr[1::2, ::2] = dctcomponent[1::8, ::8]
    dct_zr[0::2, 1::2] = dctcomponent[0::8, 1::8]
    dct_zr[1::2, 1::2] = dctcomponent[1::8, 1::8]

    return dct_zr

def dct2_8x8(component):
    r, c = component.shape
    # only keep the 1/4 lowest frequencies in each direction for the 8x8 DCT
    # Mask to set to zero the 3/4 highest frequencies
    Mr = np.ones(8)
    Mr[int(8 / 4.0):r] = np.zeros(int(3.0 / 4.0 * 8))
    Mc = Mr

    component = np.reshape(component, (-1, 8), order='C')  # after 8 pixels start a new row
    dct = sft.dct(component / 255.0, axis=1, norm='ortho')  # 2DCT applied on each of these short rows

    # apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    dct = np.dot(dct, np.diag(Mr))
    # shape it back to original shape:
    dct = np.reshape(dct, (-1, c), order='C')

    # Shape frame with columns of hight 8 by using transposition .T:
    dct = np.reshape(dct.T, (-1, 8), order='C')
    dct = sft.dct(dct, axis=1, norm='ortho')  # 2DCT applied
    # apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    dct = np.dot(dct, np.diag(Mc))
    # shape it back to original shape:
    dct = (np.reshape(dct, (-1, r), order='C')).T
    return dct

def file_size(fname):
        statinfo = os.stat(fname)
        return int(round(statinfo.st_size/1024)) #to chnage in to kilo bytes

#Software

f = open('compressed.txt', 'wb')
original = open('original.txt', 'wb')
cap = cv2.VideoCapture(0)
count = 0

#Capturing 25 frames only as mentioned in the file
for i in range(25):

    ret, frame = cap.read()
    r, c, d = frame.shape
    cv2.putText(frame, f"I Frame = {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
    cv2.imshow('Original', frame)

    Yprev = np.zeros((r, c))
    #Original RGB components stored separately
    pickle.dump(frame[:, :, 0], original)
    pickle.dump(frame[:, :, 1], original)
    pickle.dump(frame[:, :, 2], original)

    if ret == True:
        YCbCrFrame = rgb2ycbcr(frame)

        if i % 2 == 0:
            # Storing the I frame
            #print("Storing I frame")
            #Filtering the cb and cr components
            filtered_YCbCr = lowpass_filter(YCbCrFrame)
            #Downsampling
            Y, Cbd, Crd = downsampling(filtered_YCbCr)
            #Applying DCT
            ydct = dct2_8x8(Y)
            cbdct = dct2_8x8(Cbd)
            crdct = dct2_8x8(Crd)
            cv2.imshow('2D-DCT of Y', ydct)
            #Removing zeros from DCT
            ydct_zr = dct_compress_1by4(ydct)
            cv2.imshow('2D-DCT of Y zeros removed', ydct_zr)
            cbdct_zr = dct_compress_1by4(cbdct)
            crdct_zr = dct_compress_1by4(crdct)
            #Storing the I Frames after filtering, downsampling and applying DCT
            pickle.dump(ydct_zr, f)
            pickle.dump(cbdct_zr, f)
            pickle.dump(crdct_zr, f)

        else:
            # Storing only motion vectors for P frame
            #print("Storing only motion vectors for P frame")
            mv = motionvector(Y, Yprev)
            pickle.dump(mv, f)
            Yprev = Y.copy()

        cv2.imshow('Orignal Y Component', Y / 255)
        count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

print("\n\nOriginal file size: ", file_size("original.txt"), 'KB')
print("Compressed file size: ", file_size("compressed.txt"), 'KB')
#Calculating the compression factor
#print('Compression factor', file_size("original.txt") / file_size("compressed.txt"))

cap.release()
cv2.destroyAllWindows()
f.close()
original.close()