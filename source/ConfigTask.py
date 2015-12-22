'''
Configuration Task
==================

Takes a video as input and generates / saves to disk an object file as output which contains
all the configuration information.

This configuration and features are used in the next task, ImageProcessingTask to track
the microcontroller and detect LEDs

Usage:
    ConfigTask.py [<video_source>]

Keys:
    <Space Bar> - Pause the video
    c           - Clear all the marked ROI rectangles
    s           - Save the configuration file to the disk
    <Esc>       - Stop the program

----------------------------------------

'''

import numpy as np
import cv2
import pickle
import video
from collections import namedtuple
import time
# import common

class ROIselector:
    def __init__(self, win, ROI_type):
        self.win = win
        if ROI_type== 0:
            cv2.setMouseCallback(win, self.rect_circ)
        elif ROI_type== 1:
            cv2.setMouseCallback(win, self.poly_circ)
        self.drag_start = None
        self.drag_rect = None
        self.dragging_poly= None

        self.rectangles= []
        self.circles= []
        self.cNames= []
        self.cRadiuses= []
        self.polygon= []
        self.polygons= []
        self.tx0,self.ty0,self.tx1,self.ty1= 0,0,0,0
        self.counter= 0

    def poly_circ(self, event, x, y, flags, param):
        '''
        mouse callback function when user specified to use polygon as ROI
        saves polygon's vertices and circles' names and radiuses
        '''
        x, y= np.int16([x, y])
        start_point= []
        #end_point= []

        # double click event to create LEDs
        if event== cv2.EVENT_LBUTTONDBLCLK:
            x, y = np.int16([x, y])
            self.circles.append([x, y])
            #tempName= "LED-"+ str(len(self.circles))
            print("Enter the prefered name of selected LED and radius like 'name 7' without quotes: ")
            tempLED, tempRadius= raw_input().split()
            tempRadius= int(tempRadius)
            tempName= "LED-"+ str(tempLED)
            self.cNames.append(tempName)
            self.cRadiuses.append(tempRadius)
            print("Added circle centered at ("+str(x)+","+str(y)+") as '"+tempName+"' to the dataBase")

        #print(self.polygon)

        # capturing cursor position while mouse is moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_poly:
                end_point= [x, y]
                self.dragging_poly= [x, y]

        # capture left button down event to note starting point of polygon
        elif event== cv2.EVENT_LBUTTONDOWN:
            self.dragging_poly= [x, y]
            start_point= [x, y]
            if len(self.polygon)> 0:
                # starting point of next line should be around ending point of previous line
                if self.polygon[-1][0]- 10<= start_point[0]<= self.polygon[-1][0]+ 10 and self.polygon[-1][1]-10<= start_point[1]<= self.polygon[-1][1]+ 10:
                    pass
                else:
                    print("please select point near last end point")
            else:
                self.polygon.append(start_point)
                #print("start point appended")

        # capture left button up event to save the line coordinates to polygon
        elif event== cv2.EVENT_LBUTTONUP:
                self.dragging_poly= None
                end_point= [x, y]
                #self.polygon.append(end_point)

                # length of line shouldn't be less than 10 pixels values
                # added to not include a new line when we double clicked to select LED
                if not (self.polygon[-1][0]-10<= end_point[0]<= self.polygon[-1][0]+10 and self.polygon[-1][1]-10<= end_point[1]<= self.polygon[-1][1]+10):

                    # if last end point is around first point, complete the polygon and add the polygon to set of polygons
                    if self.polygon[0][0]- 10<= end_point[0]<= self.polygon[0][0]+ 10 and self.polygon[0][1]-10<= end_point[1]<= self.polygon[0][1]+10 and len(self.polygon)>= 3:
                        self.polygons.append(self.polygon)
                        print("Added polygon with "+str(self.polygons[-1])+" as points to the 'Region Of Interest' database!")
                        #print(self.polygons)
                        self.polygon= []
                        #print("polygon cleared")
                    else:
                        self.polygon.append(end_point)
                        #print(self.polygon)
                        #print("polygon endpoint appended")
                else:
                    self.polygon= self.polygon[:-1]





    def rect_circ(self, event, x, y, flags, param):
        '''
        mouse callback function when user specified to use rectangle as ROI
        saves rectangle's vertices and circles' names and radiuses
        '''
        x, y = np.int16([x, y]) # BUG
        #ch = cv2.waitKey(1)

        # capture left button down event to note one vertex of rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

        # noting the cursor position when mouse is in motion
        # starting position and cursor position will be enough to draw a rectangle
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                #print(flags, cv2.EVENT_FLAG_LBUTTON)
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
                    #self.rectangles.append([x0, y0, x1, y1])

        # capturing the double-click event to note LED name, radius, and position
        if event== cv2.EVENT_LBUTTONDBLCLK:
            x, y = np.int16([x, y])
            self.circles.append([x, y])
            #tempName= "LED-"+ str(len(self.circles))
            print("Enter the prefered name of selected LED and radius like 'name 7' without quotes: ")
            tempLED, tempRadius= raw_input().split()
            tempRadius= int(tempRadius)
            tempName= "LED-"+ str(tempLED)
            self.cNames.append(tempName)
            self.cRadiuses.append(tempRadius)
            print("Added circle centered at ("+str(x)+","+str(y)+") as '"+tempName+"' to the dataBase")

    def draw_line(self, vis, start, end):
        cv2.line(vis, tuple(start), tuple(end), (0,255,0), 2)
    def draw_polygon(self, vis, polygonx):
        for i in range(len(polygonx)- 1):
            self.draw_line(vis, polygonx[i], polygonx[i+1])
        self.draw_line(vis, polygonx[-1], polygonx[0])

    def draw(self, vis):
        '''Draws circles around LEDs and ROI(rectangle or self.polygon)'''
        if not self.drag_rect:
            # draw rectangles and circles with corresponding radius and name
            # executes when user specified polygon as ROI
            for rect in self.rectangles:
                x0,y0,x1,y1= rect
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for i in range(len(self.circles)):
                x, y= self.circles[i]
                tempR= self.cRadiuses[i]
                cv2.circle(vis, (x, y), tempR, (255,0,0), 2)
                cv2.putText(vis, self.cNames[i], (x-15, y-tempR-5),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)

            # draw polygons saved in the 'polygons' list giving a polygon as input
            for polygonx in self.polygons:
                self.draw_polygon(vis, polygonx)

            # draw lines of incomplete polygon while not drawing a new line
            for i in range(len(self.polygon)- 1):
                    self.draw_line(vis, self.polygon[i], self.polygon[i+1])

            # draw lines of incomplete polygon while drawing a new line
            if self.dragging_poly:
                #print(self.polygon, self.dragging_poly)
                for i in range(len(self.polygon)- 1):
                    self.draw_line(vis, self.polygon[i], self.polygon[i+1])
                self.draw_line(vis, self.polygon[-1], self.dragging_poly)

            return False

        # executes when user specified rectangle as ROI
        x0, y0, x1, y1= self.drag_rect
        if self.tx0== x0 and self.ty0== y0 and self.tx1== x1 and self.ty1== y1:
            self.counter+= 1
            if self.counter== 150 and [x0,y0,x1,y1] not in self.rectangles:
                # Rectangle is appended as ROI if it is unchanged for 150 frames (not in dragging position)
                self.rectangles.append([x0,y0,x1,y1])
                print("Added rectangle with ("+str(x0)+","+str(y0)+"), ("+str(x1)+","+str(y1)+") as diagonal points to 'Region Of Interest!' dataBase")
                self.counter= 0
        else:
            self.counter= 0

        self.tx0,self.ty0,self.tx1,self.ty1= x0,y0,x1,y1
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # draw rectangles and circles with corresponding radius and name
        for rect in self.rectangles:
            x0,y0,x1,y1= rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for i in range(len(self.circles)):
            x, y= self.circles[i]
            tempR= self.cRadiuses[i]
            cv2.circle(vis, (x, y), tempR, (255,0,0), 2)
            cv2.putText(vis, self.cNames[i], (x-15, y-tempR-5),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)

        return True
    @property
    def dragging(self):
        return self.drag_rect is not None




PlanarTarget = namedtuple('PlanarTarget', 'rect, keypoints, descrs, data')

class FeatureDetector:
    def __init__(self, callback):
        self.detector = cv2.ORB( nfeatures = 1000)
        self.callback= callback

    # def is_point_inside(self, r1, r2, r3, r4, r):       #useless fnction!
    #     #r1r2r4
    #     print(r1,r2,r3,r4,r4)
    #     alpha= float((r2[1] - r4[1])*(r[0] - r4[0]) + (r4[0] - r2[0])*(r[1] - r4[1])) / ((r2[1] - r4[1])*(r1[0] - r4[0]) + (r4[0] - r2[0])*(r1[1] - r4[1]))
    #     beta= float((r4[1] - r1[1])*(r[0] - r4[0]) + (r1[0] - r4[0])*(r[1] - r4[1])) / ((r2[1] - r4[1])*(r1[0] - r4[0]) + (r4[0] - r2[0])*(r1[1] - r4[1]))
    #     gamma= 1- alpha- beta
    #     if alpha >= 0 and beta >= 0 and gamma >= 0:
    #         #print(r1,r2,r3,r4,r, "dfadfs")
    #         return True
    #     #r2r3r4
    #     alpha= float((r3[1] - r4[1])*(r[0] - r4[0]) + (r4[0] - r3[0])*(r[1] - r4[1])) / ((r3[1] - r4[1])*(r2[0] - r4[0]) + (r4[0] - r3[0])*(r2[1] - r4[1]))
    #     beta= float((r4[1] - r2[1])*(r[0] - r4[0]) + (r2[0] - r4[0])*(r[1] - r4[1])) / ((r3[1] - r4[1])*(r2[0] - r4[0]) + (r4[0] - r3[0])*(r2[1] - r4[1]))
    #     gamma= 1- alpha- beta
    #     if alpha >= 0 and beta >= 0 and gamma >= 0:
    #         #print(r1,r2,r3,r4,r, "asdfasdf")
    #         return True
    #     return False

    def extract_features(self, image, ROIs, circles, user_res, ROI_type, data=None):
        '''
        extract features in a particular frame marked with ROI and LEDs
        '''
        all_ROIs_points, all_ROIs_descs, all_circles= [], [], circles

        # way to extract features when ROI type is rectangle
        if ROI_type== 0:
            all_rects= ROIs
            for rect in all_rects:
                x0, y0, x1, y1 = rect
                raw_points, raw_descrs = self.detect_features(image)

                points, descs = [], []

                for kp, desc in zip(raw_points, raw_descrs):
                    x, y = kp.pt
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        points.append(kp)
                        descs.append(desc)

                all_ROIs_points.append(points)
                all_ROIs_descs.append(descs)

            # sending the data to callback function
            self.callback(all_ROIs_points, all_ROIs_descs, all_rects, all_circles, user_res)

        # way to extract features when the ROI type is polygon
        elif ROI_type== 1:
            all_polys= ROIs
            all_polys_new= []
            all_polys_to_rects= []
            for poly in all_polys:
                #modifying polygon into a rectangle while extracting feature points!

                [[p0, q0], [p1, q1], [p2, q2], [p3, q3]]= poly[:4]
                #all_polys_new.append([[p0, q0], [p1, q1], [p2, q2], [p3, q3]])
                all_polys_new.append(poly)

                x0= int(float(poly[0][0]+ poly[3][0])/2)
                y0= int(float(poly[0][1]+ poly[1][1])/2)
                x1= int(float(poly[2][0]+ poly[1][0])/2)
                y1= int(float(poly[2][1]+ poly[3][1])/2)
                all_polys_to_rects.append([x0, y0, x1, y1])
                raw_points, raw_descrs = self.detect_features(image)

                #print(poly)
                #print(all_polys_to_rects)

                points, descs = [], []

                for kp, desc in zip(raw_points, raw_descrs):
                    x, y = kp.pt
                    if x0 <= x <= x1 and y0 <= y <= y1:                                    #first or second
                    #if self.is_point_inside([p0, q0], [p1, q1], [p2, q2], [p3, q3], [x, y]):
                        points.append(kp)
                        descs.append(desc)

                all_ROIs_points.append(points)
                all_ROIs_descs.append(descs)

            # all_polys_to_rects or all_polys_new
            # all_polys_to_rects - modified polygons into rectangles
            # all_polys_new - polygons with only four sides

            # sending the data to callback function
            self.callback(all_ROIs_points, all_ROIs_descs, all_polys_new, all_circles, user_res)


    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs


class ConfigApp:
    def __init__(self, src, ROI_type):
        self.cap = video.create_capture('test.avi')
        self.frame = None
        self.paused = False
        #self.tracker = PlaneTracker()
        self.ROI_type= ROI_type
        self.fps = 0
        cv2.namedWindow('plane')
        self.rect_sel = ROIselector('plane', ROI_type)
        self.feat_det= FeatureDetector(self.save_data)

        #CHANGE: change resolution of camera
        self.set_img_siz(user_x, user_y)

    def get_framecap(self):
        framecap = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "FPS: " + str(framecap)

    def set_img_siz(self,x,y):
        print x, y
        r1 = self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(x))
        r2 = self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(y))
        print r1, r2

    def run(self, user_x, user_y):

        frame_count = 0
        start_time = time.time()
        while True:
            playing = not self.paused and not self.rect_sel.dragging
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

            #need to select one of the following three frame resolutions

            ratio= self.frame.shape[1]/float(self.frame.shape[0])
            #self.frame= cv2.resize(self.frame, (int(user_x), int(user_x/ratio)))
            #self.frame= cv2.resize(self.frame, (int(user_x), int(user_y)))
            #self.frame= cv2.resize(self.frame, (int(user_y*ratio), int(user_y)))

            #print(self.frame.shape)
            user_res= self.frame.shape
            vis = self.frame.copy()

            self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)

            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            # if pressed 'c' clear all the marked ROIs and LEDs from the database
            if ch == ord('c'):
                self.rect_sel.rectangles= []
                self.rect_sel.circles= []
                self.rect_sel.polygons= []
                self.rect_sel.polygon= []
                self.rect_sel.cNames= []
                self.rect_sel.cRadiuses= []
                print("Cleared all marked Rectangles/Polygons & Circles from dataBase")
            # if pressed 's' save the contents of the frame to pickle file
            if ch == ord('s'):
                # save to .obj file
                # implements two main functions depends of user specified ROI type
                if self.ROI_type== 0:
                    self.feat_det.extract_features(self.frame, self.rect_sel.rectangles, self.rect_sel.circles, user_res, self.ROI_type)
                elif self.ROI_type== 1:
                    self.feat_det.extract_features(self.frame, self.rect_sel.polygons, self.rect_sel.circles, user_res, self.ROI_type)

            if ch == 27:
                break

    def save_data(self, all_ROIs_points, all_ROIs_descs, all_modified_ROIs, all_circles, user_res):
        '''
        save the data to the pickle file
        Contents of the data file:
            all_index - serialized version of feature points
            all_ROIs_descs - feature descriptors
            all_modified_ROIs - modified ROIs
            circles data:
                all_circles - circles center points
                cRadiuses - radiuses of all circles
                cNames - names of circles
            user_res - resolution specified by the user
            ROI_type - type of ROI specified by user
        '''

        #print(len(all_modified_ROIs), len(all_ROIs_points))
        file_object=  open("outputFile.p", "wb")
        #print(type(points), type(descs), type(rect))
        all_index= []

        # we couldn't pickle feature points directly
        # so we serialize it into a python array and deserialize when reading the config file in IP task
        for points in all_ROIs_points:

            index= []
            for point in points:
                temp= (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                index.append(temp)
            all_index.append(index)

        pickle.dump([all_index, all_ROIs_descs, all_modified_ROIs, [all_circles, self.rect_sel.cRadiuses, self.rect_sel.cNames], user_res, self.ROI_type], file_object)
        file_object.close()
        print("Successfully saved the whole dataBase to 'outputFile.p'")

if __name__ == '__main__':
    print __doc__
    import sys

    # take a video source from command line arguments
    try:
        video_src = 'test.avi'
    except:
        video_src = 0
    got_ans1, got_ans2= False, False

    print("Prefered resolution of the video:\nEnter two space seperated integers like '720 480' without quotes: ")

    # ask for resolution of video selected
    try:
        a= raw_input()
        #print(a)
        user_x, user_y= [int(i) for i in a.split()]
        got_ans1= True
    except:
        print("Please check the entered value and re-run the program.")

    print("Prefered ROI type:\nEnter a word- rectangle or polygon: ")

    # ask user for the prefered ROI type
    try:
        a= raw_input()
        if a== "rectangle":
            ROI_type= 0
            got_ans2= True
        elif a== "polygon":
            ROI_type= 1
            got_ans2= True
        else:
            got_ans2= False
            print("Please check the entered value and re-run the program.")
    except:
        print("Please check the entered value and re-run the program.")

    if got_ans1 and got_ans2:
        ConfigApp(video_src, ROI_type).run(user_x, user_y)
    else:
        print("Re run the program")
        sys.exit()
