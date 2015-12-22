# import the necessary packages
import imutils
import cv2
import numpy as np
from collections import namedtuple
import video
import common
from itertools import count
import pickle
import os

def split_into_rgb_channels(image):
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue

def show_hist(self):
    bin_count = self.hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(self.hist[i])
        cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist', img)

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

def enhance_light(img):  
    scale = 0.7  # whatever scale you want
    bright = -160
    img2 =  cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    frame_darker = (img_gray * scale).astype(np.uint8)
    img_gray = cv2.add(frame_darker, bright)
    ret3,img2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("AAA", opening)
    return opening

def get_color_hsv_based(hsv_value):
    color = None

    if (hsv_value >= 0 and hsv_value <=10) or (hsv_value >= 147 and hsv_value <=180):
        color = 'red'
    elif (hsv_value >= 11 and hsv_value <=32):
        color = 'orange'
    elif (hsv_value >= 33 and hsv_value <=44):
        color = 'yellow'
    elif (hsv_value >= 45 and hsv_value <=83):
        color = 'green'
    elif (hsv_value >= 84 and hsv_value <=146):
        color = 'blue'

    return color

def get_dominant_color_value(img):
    image = img
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig = image
    
    blur = cv2.GaussianBlur(img2,(5,5),0)
    ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #for c in contours:
    #    (x,y),radius = cv2.minEnclosingCircle(c)
    #    center = (int(x),int(y))
    #    radius = int(radius)
    #    cv2.circle(image,center,radius,(0,255,0),1)

    cv2.imshow('RAUl', image)

    max_height = 0
    max_hue = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    hist = cv2.calcHist( [hsv], [0], mask, [16], [0, 180] )
    #hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist = hist.reshape(-1)
    bin_count = hist.shape[0]
    bin_w = 24
    cv2.imshow('VV', img)
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(hist[i])
        cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        if h >= max_height:
            max_height = h
            max_hue = int(180.0*i/bin_count)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist', img)
    return max_hue

def get_threshold_img(img):
    #img =  cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 0.7  
    bright = -100
    gray = (gray * scale).astype(np.uint8)
    gray = cv2.add(gray, bright)
    [minVal, maxVal, minLoc, maxLoc] = cv2.minMaxLoc(gray)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    margin = 0.9
    thresh = int( maxVal * margin) # in pix value to be extracted

    #ret3,th3 = cv2.threshold(gray,thresh,maxVal,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret3,th3 = cv2.threshold(gray,thresh,maxVal,cv2.THRESH_BINARY)

    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    

    #cv2.drawContours(opening, contours, -1, (255,255,255), 1)

    approx = []
    moments = []
    for cnt in contours:
        moments.append(cv2.moments(cnt))
        #epsilon = 0.1*cv2.arcLength(cnt,True)
        approx.append(cv2.convexHull(cnt))
    
    cv2.drawContours(opening, approx, -1, (255,255,255), 1)
    return opening 




FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level = 2) #2

search_params = dict(checks=1)

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class PlaneTrackerRect:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, search_params)  # bug : need to pass empty dict (#1329)
        self.targets = []

    def add_target(self, image, vertices, keys = None, des = None, data=None):
        if keys is None:
            '''Add a new tracking target.'''
            x0, y0, x1, y1 = vertices
            raw_points, raw_descrs = self.detect_features(image)
            points, descs = [], []
            for kp, desc in zip(raw_points, raw_descrs):
                x, y = kp.pt
                if x0 <= x <= x1 and y0 <= y <= y1:
                    points.append(kp)
                    descs.append(desc)
            descs = np.uint8(descs)
            self.matcher.add([descs])
            target = PlanarTarget(image = image, rect=vertices, keypoints = points, descrs=descs, data=None)
            self.targets.append(target)
            return points, descs
        else:
            descs = np.uint8(des)
            self.matcher.add([descs])
            target = PlanarTarget(image = image, rect=vertices, keypoints = keys, descrs=descs, data=None)
            self.targets.append(target)
            return keys, descs

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame)
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

            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame, mask=None):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, mask)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

class PlaneTrackerPoly:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []

    def add_target(self, image, vertices, keys = None, des = None, data=None):
        if keys is None:
            '''Add a new tracking target.'''
            raw_points, raw_descrs = self.detect_features(image)
            points, descs = [], []
            for kp, desc in zip(raw_points, raw_descrs):
                x, y = kp.pt
                inside_poly = check_inside_poly(x,y,vertices, image)
                if inside_poly:
                    points.append(kp)
                    descs.append(desc)
            descs = np.uint8(descs)
            self.matcher.add([descs])
            target = PlanarTarget(image = image, rect=vertices, keypoints = points, descrs=descs, data=None)
            self.targets.append(target)
            return points, descs
        else:
            descs = np.uint8(des)
            self.matcher.add([descs])
            target = PlanarTarget(image = image, rect=vertices, keypoints = keys, descrs=descs, data=None)
            self.targets.append(target)
            return keys, descs

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame)
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

            array_pts = target.rect
            quad = np.float32(array_pts)
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame, mask=None):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, mask)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

def check_inside_poly(x,y,vertices, original_frame):
    img = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    size = (w, h, channels) = (h, w , 1)
    src = np.zeros(size, np.uint8)
    poly = Polygon(vertices)
    poly.draw(src, (255,255,255))
    contours,hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros(src.shape,np.float32) # array to store distances
    cnt = contours[0] # We take only one contour for testing
    is_inside = cv2.pointPolygonTest(cnt,(x,y),False)
    if is_inside == 1:
        return True
    else:
        return False

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return self.drag_rect
    @property
    def dragging(self):
        return self.drag_rect is not None

class PolySelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None

class Rectangle:
    ids = count(0)
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.id = self.ids.next()
        self.ROI = None
    def get_points(self):
        return self.pt1 + self.pt2
    def set_points(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
    def get_id(self):
        return self.id
    def clear(self):
        self.pt1 = []
        self.pt2 = []
    def draw(self, vis):
        if self.pt1:
            cv2.rectangle(vis, tuple(self.pt1), tuple(self.pt2), (0, 0, 255), 2)
            return True
    def set_ROI(self, frame):
        pt1x, pt2x, pt1y, pt2y = self.pt1[0], self.pt2[0], self.pt1[1], self.pt2[1]
        img = frame[pt1y:pt2y, pt1x:pt2x, :]
        self.ROI = img
    def get_ROI(self):
        return self.ROI

class Polygon:
    ids = count(0)
    def __init__(self, array_pts):
        self.pts = array_pts
        self.id = self.ids.next()
    def get_points(self):
        return self.pts
    def set_points(self, new_pts):
        self.pts = new_pts
    def get_id(self):
        return self.id
    def clear(self):
        self.pts = []
    def draw(self, img, color=(0,0,255)):
        if self.pts:
            pts = np.array(self.pts , np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,color,2)

class Led_Circle:
    ids = count(0)
    def __init__(self, c, r):
        self.center = c
        self.radius = r
        self.id = self.ids.next()
        self.name = 'Led '+str(self.id)
        self.color = None
        self.ROI = None
    def get_center(self):
        return self.center
    def set_center(self, c):
        self.center = c
    def get_radius(self):
        return self.radius
    def set_center(self, r):
        self.radius = r
    def get_name(self):
        return self.name
    def set_name(self, str):
        self.name = str
    def get_color(self):
        return self.color
    def set_color(self, clr):
        self.color = clr
    def get_id(self):
        return self.id
    def set_ROIs(self, frame):
        c = self.get_center()
        r = self.get_radius()
        cx, cy = c[0], c[1]
        pt1x, pt2x, pt1y, pt2y = cx - r, cx + r, cy + r, cy - r
        img = frame[pt2y:pt1y, pt1x:pt2x, :]
        self.ROI = img
    def get_ROIs(self):
        return self.ROI

class LED(Led_Circle):
    ids = count(0)
    off_line_color = (255,255,255)
    on_line_color = (255,0,0)
    def __init__(self, c, r):
        Led_Circle.__init__(self, c, r)
        self.color = None
        self.status = False
        self.frequency, self.counter = 0, 0
        self.brightness = None
        self.line_color = self.off_line_color
        self.ROI = None
        self.stackROI, self.stackColor, self.stackStatus, self.stackWhite = [], [], [], []
        self.isColorConfirmed, self.confirmedColor= False, None
        self.start, self.wait_for_reference, self.start_counting, self.finish = True, False, False, False
        self.current_status, self.reference, self.wait_finish = None, None, None
        self.whiteDensity = 0
        self.detectStatusByColor, self.colorToMatch = False, None
    def get_color(self):
        return self.color
    def set_color(self, clr):
        self.color = clr
    def get_status(self):
        return self.status
    def get_stack_info(self):
        return self.stackColor, self.stackStatus, self.stackWhite
    def confirmColor(self):
        self.isColorConfirmed = True
        self.confirmedColor = self.color
    def getColorConfirmation(self):
        return self.isColorConfirmed
    def activateStatusDetectColor(self, color):
        self.detectStatusByColor = True
        self.colorToMatch = color

    def get_led_state(self, img):
        if self.detectStatusByColor is False:
            img = enhance_light(self.get_ROI())
            total_pixels = img.size
            white = cv2.countNonZero(img)
            ratio = white / float(total_pixels)
            if ratio <= 0.1:
                return False
            else:
                return True
        else:
            if not self.color == self.colorToMatch:
                return True
            else:
                return False 

    def get_white_density(self):
        img = enhance_light(self.get_ROI())
        white = cv2.countNonZero(img)
        return white
    def set_white_density(self, p):
        self.whiteDensity = p

    def set_status(self, st):
        self.status = st
        if st is False:
            self.line_color = self.off_line_color
        else:
            self.line_color = self.on_line_color
    def get_ROI(self):
        return self.ROI
    def set_ROI_and_process(self, roi, framecap):
        self.ROI = roi
        temp = roi
        hsv_histgram_max = get_dominant_color_value(temp)
        color = get_color_hsv_based(hsv_histgram_max)
        white = self.get_white_density()
        self.set_white_density(white)
        state = self.get_led_state(roi)
        #if self.isColorConfirmed is False:
        self.color = color
        self.status = state
        if len(self.stackROI) == 10:
            self.stackROI.pop(0)
            self.stackColor.pop(0)
            self.stackStatus.pop(0)
            self.stackWhite.pop(0)
            self.stackROI.append(roi)
            self.stackColor.append(color)
            self.stackStatus.append(state)
            self.stackWhite.append(white)
        else:
            self.stackROI.append(roi)
            self.stackColor.append(color)
            self.stackStatus.append(state)
            self.stackWhite.append(white)

        if self.isColorConfirmed is True:
            if self.start is True:
                self.current_status = self.status
                self.wait_for_reference = True
                self.start = False
            if self.wait_for_reference is True:
                if not self.status == self.current_status:
                    self.reference = self.status
                    self.start_counting = True
                    self.wait_for_reference = False
            if self.start_counting is True:
                self.counter = self.counter + 1
                if not self.reference == self.status:
                    self.wait_finish = self.status
                    self.finish = True
                if self.finish is True:
                    if not self.wait_finish == self.set_status:
                        self.frequency = framecap/self.counter
                        self.counter = 0
                        self.finish = False
                        self.start_counting = False
                        self.start = True

    def draw(self, frame):
        color = self.get_color_to_draw()
        cv2.circle(frame,tuple(self.get_center()), self.get_radius(), color, 1)
        pt = self.get_center()
        x = pt[0]
        y = pt[1]
        if self.name == 'Led 4':
            y = pt[1] + self.get_radius()+15
            x = pt[0]
        elif self.name == 'Led 0' or self.name == 'Led 1' or self.name == 'Led 2':
            y = pt[1] + self.get_radius()-30
            x = pt[0]
        elif self.name == 'Led 3':
            x = x - 70
            y = y +25
        cv2.putText(frame, ""+self.name + "("+"{0:.2f}".format(self.frequency)+" Hz)", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,0),2)
    def get_id(self):
        return self.id
    def get_color_hsv_based(self, ROI):
        hsv_histgram_max = get_dominant_color_value(ROI)
        return get_color_hsv_based(hsv_histgram_max)
    def get_color_to_draw(self):
        if self.status is False:
            return (0,0,0)
        else:
            if self.isColorConfirmed is False:
                if self.color == 'red':
                    return (0,0,255)
                elif self.color == 'blue':
                    return (255,0,0)
                elif self.color == 'green':
                    return (0,255,0)
                elif self.color == 'orange':
                    return (255,106,0)
                elif self.color == 'yellow':
                    return (255,238,0)
                elif self.color == 'white':
                    return (255,255,255)
            else:
                if self.confirmedColor == 'red':
                    return (0,0,255)
                elif self.confirmedColor == 'blue':
                    return (255,0,0)
                elif self.confirmedColor == 'green':
                    return (0,255,0)
                elif self.confirmedColor == 'orange':
                    return (255,106,0)
                elif self.confirmedColor == 'yellow':
                    return (255,238,0)
                elif self.confirmedColor == 'white':
                    return (255,255,255)


    def print_info(self):
        print "LED: "+ str(self.name)
        print "Color confirmed? " + str(self.isColorConfirmed)
        print "True Color " + str(self.confirmedColor)
        print "Current color: " +str(self.color)
        print "Current status "+str(self.status)
        print "Current Frequency: " + str(self.frequency) 

class SerializationObj:
    ids = count(0)
    def __init__(self, filename):
        self.filename = os.path.join("..","IP_Server",filename)
        self.fileObject = None
    def set_filename(self, str):
        self.filename = str
    def get_filename(self):
        return self.filename
    def serialize(self, obj):
        pickle.dump(obj, self.fileObject)
    def deserialize(self):
        temp = pickle.load(self.fileObject)
        return temp
    def open_for_write(self):
        self.fileObject = open(self.filename,'wb')
    def open_for_read(self):
        self.fileObject = open(self.filename,'r')
    def close_file(self):
        self.fileObject.close()


class TrackObject:
    ids = count(0)
    def __init__(self, obj, keypoints = None, descriptors = None):
        self.obj = obj
        if isinstance(self.obj, Rectangle):
            self.tracker = PlaneTrackerRect()
        elif isinstance(self.obj, Polygon):
            self.tracker = PlaneTrackerPoly()
        self.vertices = self.obj.get_points()
        self.is_object_added = False
        self.keyPoints = keypoints
        self.descrs = descriptors
        self.keyPoints_image = None
        self.descrs_image = None

    def clear_targets(self):
        self.tracker.clear()
        self.obj.clear()

    def get_keypoints(self):
        return self.keyPoints
    def get_descriptors(self):
        return self.descrs

    def track(self, frame):
        if not self.is_object_added:
            self.keyPoints, self.descrs = self.tracker.add_target(frame, self.vertices, self.keyPoints, self.descrs)
            self.is_object_added = True
        tracked = self.tracker.track(frame)
        for tr in tracked:
            for (x, y) in np.int32(tr.p1):
                cv2.polylines(frame, [np.int32(tr.quad)], True, (0, 0, 255), 2)
                cv2.circle(frame, (x, y), 2, (255, 255, 255))

        #self.obj.draw(frame)
        self.keyPoints_image, self.descrs_image = self.tracker.detect_features(frame)





class LedDetector:
    ids = count(0)
    def __init__(self, list_LEDs_circles):
        self.LED_circles = list_LEDs_circles
        self.LEDs = []
        for led in self.LED_circles:
            self.LEDs.append(LED(led.get_center(), led.get_radius()))

    def set_ROIs(self, frame, framecap):
        for led in self.LEDs:
            c = led.get_center()
            r = led.get_radius()
            cx, cy = c[0], c[1]
            pt1x, pt2x, pt1y, pt2y = cx - r, cx + r, cy + r, cy - r
            img = frame[pt2y:pt1y, pt1x:pt2x, :]
            led.set_ROI_and_process(img, framecap)

    def check_led_state_and_color(self):
        for led in self.LEDs:
            if led.getColorConfirmation() is False:
                color, state, white = led.get_stack_info()
                if len(color) > 3:

                    if state[-1] is True and state[-2] is False:
                        if not color[-1] == color[-2]:
                            led.confirmColor()
                        elif color[-1] == color[-2]:
                            led.confirmColor()

                    if state[-1] is True and state[-2] is True:
                        if not color[-1] == color[-2]:
                            if white[-1] > 1.1*white[-2]:
                                led.set_color(color[-2])
                                led.confirmColor()
                                led.activateStatusDetectColor(color[-2])
                            elif white[-1]*1.1 < white[-2]:
                                led.set_color(color[-1])
                                led.confirmColor()
                                led.activateStatusDetectColor(color[-1])
                            elif color[-1] == 'green':
                                led.set_color(color[-2])
                                led.confirmColor()
                                led.activateStatusDetectColor(color[-1])
                            elif color[-2] == 'green':
                                led.set_color(color[-1])
                                led.confirmColor()
                                led.activateStatusDetectColor(color[-2])


    def get_LEDs(self):
        return self.LEDs
    def set_LEDs(self, LEDs_list):
        self.LEDs = LEDs_list

    def draw(self, frame):
        for led in self.LEDs:
            led.draw(frame)

    def print_txt(self):
        for led in self.LEDs:
            led.print_info()

    def detect(self, frame, framecap):
        self.set_ROIs(frame, framecap)
        self.check_led_state_and_color()
        self.draw(frame)
        #self.print_txt()

        
