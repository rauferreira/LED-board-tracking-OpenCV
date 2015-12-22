#import sys
#sys.path.append("C:\\opencv\\build\\python\\2.7")
#import cv2
#import cv2.cv as cv
#import time


#capture = cv2.VideoCapture(0)

#num_frame = 0


#size = capture.get(cv.CV_CAP_PROP_FRAME_WIDTH), capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

#size_new = capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320),capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

##size_new = capture.get(cv.CV_CAP_PROP_FRAME_WIDTH), capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)




#start = time.time()

#while(True):
#    ret, frame = capture.read()
#    if num_frame < 60:
#        num_frame = num_frame + 1
#    else:
#        break

#total_time = (time.time() - start)
#fps = (num_frame / total_time)
#print str(num_frame) + ' frames ' + str(total_time) + ' second = ' + str(fps) + ' fps'




#capture.release()
#cv2.destroyAllWindows()


