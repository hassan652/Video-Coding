import numpy as np
import cv2
import pickle
import scipy.signal

cap = cv2.VideoCapture(0)
 
N=2
#scaling_factorX=0.5
#scaling_factorY=0.5
o = open('OriginalVideo.txt', 'wb')
fw = open('Framewise.txt', 'wb')

#Lowpass Filter
filt1=np.ones((N,N))/N
filt2=scipy.signal.convolve2d(filt1,filt1)/N


retval, frame= cap.read()
[rows,columns,d] = frame.shape #print rows,columns
cropped= frame[int(rows/2)-120:int(rows/2)+120 , int(columns/2)-160:int(columns/2)+160, :] #frame cropped from center
r_cr, c_cr, d_cr = cropped.shape

Yprev=np.zeros((rows,columns))#from Y of previous frame

framevectors=np.zeros((rows,columns,3))#vectors for current frame

mv=np.zeros((int(rows/8),int(columns/8),2))#motion vectors for each block

#processing 25 frames
for n in range(25):

    print("Frame no. ",n)
    ret, frame = cap.read()
    
    rows,columns,c=frame.shape
    #frame = cv2.resize(frame, (320, 240)
    #cap.set(cv2.CAP_PROP_FPS,30)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

    if ret==True:
        
        redd = np.zeros((rows,columns,c))

        Y=(0.114*frame[:,:,0]+0.587*frame[:,:,1]+0.299*frame[:,:,2])
        Cb=(0.4997*frame[:,:,0]-0.33107*frame[:,:,1]-0.16864*frame[:,:,2])
        Cr=(-0.081282*frame[:,:,0]-0.418531*frame[:,:,1]+0.499813*frame[:,:,2])
        #components stored in reduced array
        redd[:,:,0]=Y
        redd[:,:,1]=Cb
        redd[:,:,2]=Cr

        #frame=cv2.resize(frame,None,fx=scaling_factorx,fy=scaling_factory,interpolation=cv2.INTER_AREA)

        #displaying normalized frame and motion vectors
        cv2.imshow('Original',frame/255.0+framevectors)
        #filtering colorcomponents before downsampling
        Crfilt=scipy.signal.convolve2d(Cr,filt2,mode='same')
        Cbfilt=scipy.signal.convolve2d(Cb,filt2,mode='same')

        DCr=Crfilt[0::N,::N]
        DCb=Cbfilt[0::N,::N]

        block = [8,8]

        framevectors=np.zeros((rows,columns,3))
        #motion estimation
        #for yblock in range(int((rows-8)/8)):
        for yblock in range(28):
            block[0]=yblock*8+8

            #for xblock in range(int((columns-8)/8)):#(640-8)/8
            for xblock in range(38):
                block[1]=xblock*8+8

                #current block
                Yc=Y[block[0]:block[0]+8 ,block[1]:block[1]+8]                      #Current frame
                #previous block
                Yp=Yprev[block[0]-8 : block[0]+8, block[1]-8 : block[1]+8]          #Previous frame
                #high value for mean absolute error for initialization
                bestmae=10000.0

                #'''
                for ymv in range(-8,8):
                    for xmv in range(-8,8):
                        diff = Yc - Yprev[block[0] + ymv: block[0] + ymv + 8, block[1] + xmv: block[1] + xmv + 8];
                        mae = sum(sum(np.abs(diff))) / 64
                        if mae < bestmae:
                            bestmae = mae
                            mv[yblock, xblock, 0] = ymv
                            mv[yblock, xblock, 1] = xmv
                tx = block[1]
                ty = block[0]

                #'''

                Ycorr=scipy.signal.fftconvolve(Yp, Yc[::-1,::-1], mode='valid')
                index2d = np.unravel_index(np.argmax(Ycorr),(Ycorr.shape))
                secarg = np.add(block,index2d)
                
                cv2.line(framevectors, tuple(block)[::-1], tuple(secarg)[::-1],(0.0,1.0,0.0),1)
                #display motion vector blocks in lines
                #cv2.line(framevectors, (block[1], block[0]),(block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int)),(1.0, 1.0, 1.0))

        Yprev=Y.copy()

        frame=np.array(frame,dtype='uint8')
        Y=np.array(Y, dtype='uint8')
        DCr=np.array(DCr, dtype='int8')
        DCb=np.array(DCb, dtype='int8')
        pickle.dump(frame,fw)
          
        pickle.dump(Y,o)
        pickle.dump(DCr,o)
        pickle.dump(DCb,o)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break

    
a = fw.tell()/1024.0**2
b = o.tell()/1024.0**2

print ('File size framewise:',a ,'Mb')

print ('File size after applying motion vectors:',b ,'Mb')
print ('Compression factor:',(a/b))   
print ('Space saving:',((1-(b/a))*100.0 ),'%')
cap.release()
o.close()
fw.close()
cv2.destroyAllWindows()
