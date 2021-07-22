import numpy as np
import cv2
import sys
import pickle
import scipy.signal

cap = cv2.VideoCapture(0)
# Get size of frame:
ret, frame = cap.read()
ro1, co1, d1 = frame.shape
# Cropped frame as per the requirements
cropped = frame[int(ro1 / 2) - 120:int(ro1/ 2) + 120, int(co1 / 2) - 160:int(co1 / 2) + 160,:]
rows, columns, c = cropped.shape

# Prevous Y frame:
Yprev = np.zeros((rows, columns))

# motion vectors, for each block a 2-d vector:
mv = np.zeros((int(rows / 8), int(columns / 8), 2))

#processing 25 frames
for n in range(25):

    print("Frame no. ",n)
    ret, frame = cap.read()
    rows, columns, c = frame.shape
    #frame cropped from center as per the requirements
    cropped = frame[int(rows / 2) - 120:int(rows / 2) + 120, int(columns / 2) - 160:int(columns / 2) + 160,:]
    r_cropped, col_cropped, c_cropped = cropped.shape
    #RGB to YCbCr
    b = cropped[:, :, 0]
    g = cropped[:, :, 1]
    r = cropped[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 127 + (-0.16864 * r - 0.33107 * g + 0.49970 * b)
    cr = 127 + (0.499813 * r - 0.418531 * g - 0.081282 * b)
    ycbcr = np.zeros(cropped.shape)
    ycbcr[:, :, 0] = y
    ycbcr[:, :, 1] = cb
    ycbcr[:, :, 2] = cr

    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    # Start pixel for block wise motion estimation:
    block = [8, 8]

    # Vectors for current frame as graphic:
    framevectors = np.zeros((r_cropped, col_cropped, 3))

    # for loops for the blocks, determines the no. of motion vectors #24 blocks(8x8p) in y direction
    for yblock in range(24):
        block[0] = yblock * 8 + 10
        # 30 blocks(8x8p) in x direction
        for xblock in range(32):
            block[1] = xblock * 8 + 10
            Y_current = Y[block[0]:block[0] + 8, block[1]:block[1] + 8]  # vertical , horizontal

            bestmae = 10000.0    # best mean absolute error (any high value)

            # For loops for the motion vector, full search at +-8 integer pixels
            # y component of the motion vector
            for ymv in range(-8, 8):
                # x component of the motion vector
                for xmv in range(-8, 8):
                    diff = Y_current - Yprev[block[0] + ymv: block[0] + ymv + 8, block[1] + xmv: block[1] + xmv + 8]
                    mae = sum(sum(np.abs(diff))) / 64

                    if mae < bestmae:
                        bestmae = mae
                        mv[yblock, xblock, 0] = ymv
                        mv[yblock, xblock, 1] = xmv
            if (mae > 20):
                start_point = (block[1], block[0])
                end_point = (block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int))
                color = (0, 255, 0)
                thickness = 1
                tipLength = 0.5
                cv2.arrowedLine(framevectors, start_point, end_point, color, thickness, tipLength= tipLength)

    cv2.imshow('Max Resolution', frame)
    cv2.imshow('320 * 240 Frame used for processing', cropped / 255.0 + framevectors)

    Yprev = Y.copy()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()