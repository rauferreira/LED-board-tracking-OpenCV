'''
Image Processing Task
=====================

Takes the object file created in Config task and tracks for the ROI in mentioned
video. Tracks LEDs and calculates their status, color, blinking frequency as output.

It sends this data to Interpreter task which runs on another computer in the same
network via ethernet. 

Usage:
    ImageProcessingTask.py [<saved/object/file> [<video_source>]]

Keys:
    <Space Bar> - Pause the video
    <Esc>       - Stop the program

----------------------------------------

'''

import numpy as np
import time
import cv2
import pickle
import video
from collections import namedtuple
import colorsys
import socket
import server
import threading

#DATA= None

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')


def send_data2(data):
    '''gets data from IP task'''
    global DATA
    DATA= data
    parse_data(DATA)

 
def start_server2():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('127.0.0.1', 8607)
    print "Starting server on", server_address, "..."

    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    print "Waiting for connection..."
    connection, client_address = sock.accept()
    print "Connection from: ", client_address

    return sock, connection


class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []
        self.user_res= []
        self.ROI_type= 0
        #self.all_circles
        #self.all_cNames
        #self.all_cRadiuses
        self.all_circles_new= []

    def load_data(self, file_name, data=None):
        '''loads data from the pickle file'''
        try:
            input_file= open(file_name, "r")
            #print(file_name)
        except:
            print("Unable to open the file- "+file_name+". Please re-run the program.")
        [all_index, all_rects_descs, all_rects, [self.all_circles, self.all_cRadiuses, self.all_cNames], self.user_res, self.ROI_type]= pickle.load(input_file)

        # deserializing the contents of 'indes' into feature points
        for i in range(len(all_rects)):
            index= all_index[i]
            descs= all_rects_descs[i]
            rect= all_rects[i]

            points= []
            for point in index:
                temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
                points.append(temp)

            descs = np.uint8(descs)
            self.matcher.add([descs])
            target = PlanarTarget(rect=rect, keypoints = points, descrs=descs, data=None)
            self.targets.append(target)


    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame)

        # see if no.of feature points is greater than our MIN_MATCH_COUNT
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            # creating quad based on user specified ROI type
            if self.ROI_type== 0:
                x0, y0, x1, y1 = target.rect
                quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            elif self.ROI_type== 1:
                #x0, y0, x1, y1 = target.rect
                #quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])         #for method 1 or method 2
                quad = np.float32(target.rect)

            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            # transforming saved led positions to new positions based on new quad shape!
            self.all_circles_new= []
            for circleI in range(len(self.all_circles)):
                new_point= cv2.perspectiveTransform(np.float32(self.all_circles[circleI]).reshape(1, -1, 2), H).reshape(-1,2)
                #print(self.all_circles[circleI], point)
                self.all_circles_new.append(new_point)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

class ledApp:
    def __init__(self):
        #data of all the leds
        self.statuses= None     #False= off; True= on
        self.max_leds= 10
        self.colors_name= None
        self.colors_rgb= None
        self.frequencies= [-1 for i in range(self.max_leds)]
        self.frame= None
        self.blur= None
        self.gray= None
        self.thresholded= None
        self.names= None
        self.radiuses= None
        self.timeStamps= [[None, None] for i in range(self.max_leds)]
        self.flag_first= [1 for i in range(self.max_leds)]

    def starter(self, frame, circles, names, radiuses, timeStamp):
        #start process with remaining functions
        #print(len(circles))
        self.frame= frame
        self.blur = cv2.GaussianBlur(self.frame,(5,5),0)
        self.gray= cv2.cvtColor(self.blur, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("plane2", self.gray)
        #cv2.imshow("blur", self.blur)

        self.radiuses= radiuses
        self.names= names

        self.statuses= []
        self.colors_name= []
        self.colors_rgb= []
        self.brightness = []

        # print(self.frequencies)
        # print(self.timeStamps)
        # detect status, frequency, color for every circle in each frame
        for i in range(len(circles)):
            #print
            #print("circle: "+ names[i])
            temp_status, temp_color, temp_rgb, temp_brigth = self.get_status_color_freq(circles[i][0], self.radiuses[i], self.names[i], timeStamp)
            #print(temp_status, temp_color)
            
            self.statuses.append(temp_status)
            self.colors_name.append(temp_color)
            self.colors_rgb.append(temp_rgb)
            self.brightness.append(temp_brigth)


    def get_status_color_freq(self, circle, radius, name, timeStamp):
        '''detects status, color, frequency of an LED'''
        ret= []
        x, y= circle
        y,x= int(x), int(y)

        #status detector
        '''change with respectively to radius obtained'''
        th, self.thresholded= cv2.threshold(self.gray[x-radius-3:x+radius+4, y-radius-3:y+radius+4], 240, 255, cv2.THRESH_BINARY)
        #cv2.imshow("plane3", self.thresholded)

        area_sum= sum(sum(self.thresholded)) #max= 7*255= 1785        #threshold value = 240
        #print(self.thresholded[x-3:x+4, y-3:y+4])
        #print(area_sum)

        # status of LED is based on area sum of thresholded frame
        if(area_sum>= 1000):
            ret.append(True)
        else:
            ret.append(False)

        
        circle_index= self.names.index(name)
        #color detector         values in BGR format

        brightness = 0

        #print(ret[0], name)
        if(ret[0]== True):      #only if the status is on!

            # select small region with padding around user specified radius of circle
            if self.flag_first[circle_index]== 1:
                # print(name+ " switched on at "+ str(timeStamp))
                # print(name + " first switched on position "+ str(self.timeStamps[circle_index][0]))
                # print(name+ " last switched off position "+ str(self.timeStamps[circle_index][1]))
                
                if self.timeStamps[circle_index][0]!= None and self.timeStamps[circle_index][1]!= None:
                    self.frequencies[circle_index]= 1000/abs(self.timeStamps[circle_index][1]- self.timeStamps[circle_index][0])
                    #print(self.frequencies[circle_index])
                elif self.timeStamps[circle_index][0]== None:
                    self.frequencies[circle_index]= -2 # hasn't switched off yet!

                self.timeStamps[circle_index][0]= timeStamp
                self.flag_first[circle_index]= 0

            
            rgb_small= self.blur[x-radius-3:x+radius+4, y-radius-3:y+radius+4]
            gray_small= self.gray[x-radius-3:x+radius+4, y-radius-3:y+radius+4]
            threshold_small= self.thresholded
            hsv_small= cv2.cvtColor(rgb_small, cv2.COLOR_BGR2HSV)

            threshold1_small= np.array(list(gray_small))
            
            
            #cv2.imshow(name+ " rgb", rgb_small)
            #cv2.imshow(name+ " gray", gray_small)
            #cv2.imshow(name+ " threshold", threshold_small)
            
            #cv2.imshow(name+ " hsv", hsv_small)
            
            # all color categories
            color_names= ["orange", "yellow", "green", "cyan", "blue", "red"]
            color_pixels= [0, 0, 0, 0, 0, 0]

            counter = 0
            # uses hsv value of a pixel to detect the color of LED
            for i in range(len(rgb_small)):
                for j in range(len(rgb_small[0])):
                    
                    # take pixels within this range into consideration when detecting color of LED
                    if gray_small[i][j]<= 240 and gray_small[i][j]>= 150:
                        threshold1_small[i][j]= 255
                    else:
                        threshold1_small[i][j]= 0

                    # excluding pixels with high brightness
                    # and excluding pixels with high brightness and low brightness

                    if threshold_small[i][j]== 255 or threshold1_small[i][j]== 0:
                    #if threshold_small[i][j]== 255:
                        rgb_small[i][j]= ([255, 255, 255])
                        brightness = brightness + hsv_small[i][j][2]
                        counter = counter + 1
                        #print(hsv_small[i][j])
                    else:
                        #change it to hsv, detect color, add to array,
                        temp_h= hsv_small[i][j][0]
                        brightness = brightness + hsv_small[i][j][2]
                        counter = counter + 1
                        if temp_h<= 20:
                            # orange
                            color_pixels[0]+= 1
                        elif temp_h<= 38:
                            # yellow
                            color_pixels[1]+= 1
                        elif temp_h<= 80:
                            # green
                            color_pixels[2]+= 1
                        elif temp_h<= 105:
                            # cyan
                            color_pixels[3]+= 1
                        elif temp_h<= 135:
                            # blue
                            color_pixels[4]+= 1
                        elif temp_h> 135:
                            # red
                            color_pixels[5]+= 1

            ret.append(color_names[color_pixels.index(max(color_pixels))])
            ret.append(sum(sum(rgb_small)))
            #print(ret[-1], name)

            #cv2.imshow(name+ " rgb modified", rgb_small)
            #cv2.imshow(name+ " threshold1", threshold1_small)
            brightness = brightness / counter
            ret.append(brightness)
        # is status of LED is off, append None
        else:
            self.flag_first[circle_index]= 1
            self.timeStamps[circle_index][1]= timeStamp
            ret.append(None)
            ret.append(None)
            brightness = 0
            ret.append(brightness)
        
        return ret


class ImageProcessionApp:
    def __init__(self, file_name, src):
        self.cap = video.create_capture(src)
        self.frame = None
        self.paused = False
        self.fps= self.cap.get(5)
        self.file_name= file_name
        self.tracker = PlaneTracker()
        self.ledModifier= ledApp()
        global DATA
        cv2.namedWindow('plane')

    def set_img_siz(self,x,y):
        r1 = self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(x))
        r2 = self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(y))

    def get_framecap (self):
        pass

    def run(self):
        self.tracker.load_data(self.file_name)
        self.set_img_siz(self.tracker.user_res[1], self.tracker.user_res[0])

        DATA_OLD= None

        frame_count = 0
        start_time = time.time()
        while True:
            playing = not self.paused
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                frame_count = frame_count + 1
                if not ret:
                    break
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    self.fps = frame_count/elapsed
                    print "Current FPS: " + str(self.fps)
                    start_time = time.time()
                    frame_count = 1
                self.frame = frame.copy()
            

            # print
            # print("Another frame................")
            vis = self.frame.copy()
            
            tracked = self.tracker.track(self.frame)


            #send to ledApp to know statuses of leds
            timeStamp= time.time() -  start_time
            self.ledModifier.starter(vis, self.tracker.all_circles_new, self.tracker.all_cNames, self.tracker.all_cRadiuses, timeStamp)

            # print(self.ledModifier.names)
            # print(self.ledModifier.statuses)
            # print(self.ledModifier.colors_name)
            # print(self.ledModifier.colors_rgb)
            # print(self.ledModifier.frequencies)



            # show tracked lines
            for tr in tracked:
                #print(tr.quad)
                cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                # for (x, y) in np.int32(tr.p1):
                #     cv2.circle(vis, (x, y), 2, (255, 255, 255))
                #print(tr.circles)
            
            # show tracked circles
            for i in range(len(self.tracker.all_circles_new)):
                #print(new_center)
                [x, y]= np.int32(self.tracker.all_circles_new[i][0])
                tempR= self.tracker.all_cRadiuses[i]
                cv2.circle(vis, (x, y), tempR, (255,0,0), 2)
                cv2.putText(vis, self.tracker.all_cNames[i], (x-15, y-tempR-5),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)
            if frame_count == 1:
                cv2.imwrite('configuration_overview.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                        # use the lists created in ledapp to send to interpreter/server task!
            DATA= [self.ledModifier.names, self.ledModifier.statuses, self.ledModifier.colors_name, 
                    self.ledModifier.colors_rgb, self.ledModifier.frequencies, self.ledModifier.brightness, self.fps, vis]
            
            # taking last known values of  color_name and color_rgb if they are None
            if DATA_OLD!= None:
                for x in range(2, 4):
                    for y in range(len(DATA[x])):
                        if DATA[x][y]== None:
                            DATA[x][y]= DATA_OLD[x][y]

            DATA_OLD= DATA

            server.send_data(DATA)
            #print(DATA)

            
            cv2.imshow('plane', vis)
            
            #print("here1")
            ch = cv2.waitKey(1)
            #print("here2")
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == 27:
                break



class myThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    def run(self):
        print "Starting " + self.name
        # Get lock to synchronize threads
        # threadLock.acquire()
        if self.name== "Image Processing":
            ImageProcessionApp(file_name, video_src).run()
        if self.name== "server":
            server.run()
        # Free lock to release next thread
        # threadLock.release()



if __name__ == '__main__':
    print __doc__
    import sys
    
    try: 
        video_src = 'test.avi'
    except: 
        video_src = 0


    # start two threads - server thread and IP thread
    threadLock=threading.Lock()
    threads= []

    print("\n\nWaiting for the configuration object PATH...")
    sock, connection= start_server2()
    command = connection.recv(128)
    print command
    print "Path received! Starting..."
    command = command.split(":")
    command = command[1]
    connection.close()
    sock.close()

    t1= myThread(2, "server")
    #t2= threading.Thread(name= "Image Processing", target= ImageProcessionApp(file_name, video_src).run)
    t2= myThread(1, "Image Processing")
    #ImageProcessionApp(file_name, video_src).run()
    
    t1.start()

    file_name = command

    t2.start()