import cv2
import numpy as np
import pickle
import scipy.signal as sp

#Functions

def RGB2YCBCR(frame):
    R = frame[:, :, 2]
    G = frame[:, :, 1]
    B = frame[:, :, 0]

    Y = (0.299 * R + 0.587 * G + 0.114 * B)
    Cb = (-0.16864 * R - 0.33107 * G + 0.49970 * B) + 128
    Cr = (0.499813 * R - 0.418531 * G - 0.081282 * B) + 128
    return Y, Cb, Cr
def chromaSubSampleWithZero(N, Cb, Cr):
    subCb = np.zeros(Cb.shape)
    subCb[0::N, 0::N] = Cb[0::N, 0::N]

    subCr = np.zeros(Cr.shape)
    subCr[0::N, 0::N] = Cr[0::N, 0::N]
    return subCb, subCr
def chromaSubSampleWithoutZero(N, Cb, Cr):
    subCb = Cb[0::N, 0::N]
    subCr = Cr[0::N, 0::N]
    return subCb, subCr

#cap object
cap = cv2.VideoCapture(0)
originalFile = open('videorecord.txt', 'wb')
reducedFile = open('videorecord_DS.txt', 'wb')
reducedFileCompressed = open('videorecord_DS_compressed.txt', 'wb')
#sampling factor
N = 4

#For 25 frames
for i in range(25):
#while True:
    ret, frame = cap.read()
    if ret == True:
        #Separating R,G,B components
        R = frame[:, :, 2]
        G = frame[:, :, 1]
        B = frame[:, :, 0]
        #Y, Cb, Cr components from R, G, B
        Y = (0.299*R + 0.587*G + 0.114*B)
        Cb = (-0.16864*R - 0.33107*G + 0.49970*B)
        Cr = (+0.499813*R - 0.418531*G - 0.081282*B)

        #Filters
        #Defining filters
        rectF = np.ones((N, N)) / N
        pyraF = sp.convolve2d(rectF, rectF) / N
        #Pyramid LP filtered Chroma components
        CbP_lp = sp.convolve2d(Cb, pyraF, mode='same')
        CrP_lp = sp.convolve2d(Cr, pyraF, mode='same')
        cv2.imshow('Pyramid Low Pass Filter Cb', CbP_lp / 255)
        cv2.imshow('Pyramid Low Pass Filter Cr ', CrP_lp / 255)

        #Rect LP filtered Chroma components
        CbR_lp = sp.convolve2d(Cb, rectF, mode='same')
        CrR_lp = sp.convolve2d(Cr, rectF, mode='same')
        cv2.imshow('Rectangle Low Pass Filter Cb ', CbR_lp/255)
        cv2.imshow('Rectangle Low Pass Filter Cr ', CrR_lp/255)

        #Downsampling
        #Downsampling with and without zeros
        cbds0, crds0 = chromaSubSampleWithZero(N, CbP_lp, CrP_lp)
        cbds, crds = chromaSubSampleWithoutZero(N, CbP_lp, CrP_lp)

        #Displaying DS components with and without zeros
        cv2.imshow('Chrominance - Cb with zeros', np.abs(cbds0 / 255.))
        cv2.imshow('Chrominance  - Cr with zeros', np.abs(crds0 / 255.))
        cv2.imshow('Chrominance - Cb without zeros', np.abs(cbds / 255.))
        cv2.imshow('Chrominance  - Cr without zeros', np.abs(crds / 255.))

        #Dumping original, subsampled with zeros and subsampled without zeros in respective files
        reduced = frame.copy()
        enc = np.zeros(reduced.shape)
        enc1 = np.zeros(reduced.shape)

        #Original File
        enc[:, :, 0] = Y
        enc[:, :, 1] = Cb
        enc[:, :, 2] = Cr
        pickle.dump(enc, originalFile)

        enc2 = np.zeros(reduced.shape)
        #Downsampled with zeros
        enc2[:, :, 0] = Y
        enc2[:, :, 1] = cbds0
        enc2[:, :, 2] = crds0
        pickle.dump(enc2, reducedFile)
       #pickle.dump(enc[:, :, 0], reducedFile)
        #pickle.dump(enc[0::2, 0::2, 1], reducedFile)
        ##ickle.dump(enc[0::2, 0::2, 2], reducedFile)

        #Subsampled without zeros
        pickle.dump(Y, reducedFileCompressed)
        pickle.dump(cbds, reducedFileCompressed)
        pickle.dump(crds, reducedFileCompressed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

Reduced_F = reducedFile.tell() / 1024.0 ** 2
Original_F = originalFile.tell() / 1024.0 ** 2
Reduced_Compressed = reducedFileCompressed.tell() / 1024.0 ** 2
print('Actual file size : ', Original_F, 'Mb')
print('Reduced file size: ', Reduced_F, 'Mb')
print('Reduced file size compressed: ', Reduced_Compressed, 'Mb')
print('Compression factor Actual vs Reduced with zeros')
print(Reduced_Compressed / Reduced_F)  # Uncompressed data/ Compressed data
print('Space saving, Actual vs Reduced with zeros')
print((1 - (Reduced_Compressed / Reduced_F)) * 100.0, 'percent')
print('Compression factor Reduced with zeros Vs Reduced without zeros')
print(Reduced_F / Original_F)
print('Space saving, Reduced with zeros Vs Reduced without zeros')
print((1 - (Reduced_F / Original_F)) * 100.0, 'percent')

cap.release()
cv2.destroyAllWindows()
originalFile.close()
reducedFile.close()
reducedFileCompressed.close()
cap.release()