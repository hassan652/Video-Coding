import numpy as np
import cv2
import pickle
import scipy.signal as sp


def upSample(downSampled,N):
    c = downSampled.shape[1]
    r = downSampled.shape[0]
    sampleFrame = np.zeros((r * N, c * N))
    sampleFrame[0::N,0::N] = downSampled
    return sampleFrame

def sampler420(frame):
    """
    frame: 2D component(for example Cb or Cr)
    N:     Upsampling factor for 420 scheme.
           Should be selected based on Downsampling factor
    """
    r, c = frame.shape
    sampleFrame = np.zeros((r*2,c*2))
    sampleFrame[::2, ::2] = frame
    sampleFrame[1::2,] = sampleFrame[::2,]
    sampleFrame[:,1::2] = sampleFrame[:,::2]
    return sampleFrame

def ycbcr2rgb(frame):

    Y = (frame[:,:,0])/255.
    Cb = (frame[:,:,1])/255.
    Cr = (frame[:,:,2])/255.

    '''Compute RGB components'''
    R = (0.771996*Y -0.404257*Cb + 1.4025*Cr)
    G = (1.11613*Y - 0.138425*Cb - 0.7144*Cr)
    B = (1.0*Y + 1.7731*Cb)

    '''Display RGB Components'''
    RGBFrame = np.zeros(frame.shape)
    RGBFrame[:,:,2] = R
    RGBFrame[:,:,1] = G
    RGBFrame[:,:,0] = B

    return RGBFrame


#sampling factor
N = 4
#Defining filters
rectf = np.ones((N,N))/N
pyraf = sp.convolve2d(rectf,rectf)/N
f = open('videorecord_DS_compressed.txt', 'rb')
try:
    ctr = 1
    while True:
        #Loading Data from file
        Y = pickle.load(f)
        Y = Y.astype(np.uint8)
        r, c = Y.shape

        #Extracting Cb and Cr components from the file
        downCb = pickle.load(f)
        downCr = pickle.load(f)

        #Upsampling using defined function
        upCb = upSample(downCb, N)
        upCr = upSample(downCr, N)

        #Filtering
        #Rect filter
        rectfY = sp.convolve2d(Y, rectf, mode='same')
        rectfCb = sp.convolve2d(upCb, rectf, mode='same')
        rectfCr = sp.convolve2d(upCr, rectf, mode='same')
        #Pyramid filter
        pyrafY = sp.convolve2d(Y, pyraf, mode='same')
        pyrafCb = sp.convolve2d(upCb, pyraf, mode='same')
        pyrafCr = sp.convolve2d(upCr, pyraf, mode='same')

        frame = np.zeros((r, c, 3))
        frame[:, :, 0] = Y
        frame[:, :, 1] = upCb
        frame[:, :, 2] = upCr

        rectFrame = np.zeros((r, c, 3))
        rectFrame[:, :, 0] = Y
        rectFrame[:, :, 1] = rectfCr
        rectFrame[:, :, 2] = rectfCb

        pyraFrame = np.zeros((r, c, 3))
        pyraFrame[:, :, 0] = Y
        pyraFrame[:, :, 1] = pyrafCr
        pyraFrame[:, :, 2] = pyrafCb

        #Decoding and Displaying Reconstructed R, G, B frame
        reduced = np.zeros((r, c, 3))
        reduced[:, :, 0] = Y
        reduced[:, :, 1] = upCb
        reduced[:, :, 2] = upCr

        #Y, Cb, Cr components
        framedec = reduced.copy()
        Y = (framedec[:, :, 0]).astype(np.uint8) / 255.
        Cb = (framedec[:, :, 1]) / 255.
        Cr = (framedec[:, :, 2]) / 255.
        R = (0.771996 * Y - 0.404257 * pyrafCb + 1.4025 * pyrafCr)
        G = (1.11613 * Y - 0.138425 * pyrafCb - 0.7144 * pyrafCr)
        B = (1.0 * Y + 1.7731 * pyrafCb)

        frame = np.zeros(framedec.shape)
        frame[:, :, 2] = R
        frame[:, :, 1] = G
        frame[:, :, 0] = B
        # Separate reconstructed R, G, B components
        #cv2.imshow('Red', R)
        #cv2.imshow('Green', G)
        #cv2.imshow('Blue', B)

        #Simple reconstructed RGB frame
        cv2.imshow("RGB Reconstructed", frame)

        #Filtered Reconstructed RGB frames
        #cv2.imshow('RGB with Rect filter', ycbcr2rgb(rectFrame))
        #cv2.imshow('RGB with Pyramid Filter', ycbcr2rgb(pyraFrame))

        if cv2.waitKey(400) & 0xFF == ord('q'):
            break
except (EOFError):
    pass


#cv2.namedWindow('RGB with Rect filter', cv2.WINDOW_NORMAL)
#cv2.namedWindow('RGB with Pyramid Filter', cv2.WINDOW_NORMAL)



















'''
    try:
        while True:
            #Conversion of string back to Matrix/Tensor after de-pickling from file f
            Y = pickle.load(f)
            Y = Y.astype(np.uint8)
            r, c = Y.shape

            downCb = pickle.load(f)
            downCr = pickle.load(f)

            #Upsampling using defined function
            upCb = sampler420(downCb)
            upCr = sampler420(downCr)

            #Defining Y, Cb, Cr for reconstructed frame
            reduced = np.zeros((r, c, 3))

            reduced[:, :, 0] = Y
            reduced[:, :, 1] = upCb
            reduced[:, :, 2] = upCr

            #Decoding
            framedec = reduced.copy()
            Y = (framedec[:, :, 0]).astype(np.uint8) / 255.
            Cb = (framedec[:, :, 1]) / 255.
            Cr = (framedec[:, :, 2]) / 255.

            #Defining R, G, B components for the reconstructed frame using respective Y, Cb, Cr values
            R = (0.771996 * Y - 0.404257 * Cb + 1.4025 * Cr)
            G = (1.11613 * Y - 0.138425 * Cb - 0.7144 * Cr)
            B = (1.0 * Y + 1.7731 * Cb)

            cv2.imshow('Red', R)
            cv2.imshow('Green', G)
            cv2.imshow('Blue', B)

            #Combining the frame, and displaying it
            frame = np.zeros(framedec.shape)
            frame[:, :, 2] = R
            frame[:, :, 1] = G
            frame[:, :, 0] = B

            cv2.imshow("Colored Reconstructed", frame)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    except (EOFError):
        pass

cv2.destroyAllWindows()
'''