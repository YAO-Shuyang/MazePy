'''
This file provides a useful tool to manually select the edge of the environment.
This file was first designed by Ang Li, the original link was listed here: 
https://github.com/AmazingAng/maze_learning/blob/main/Behav_maze.py

The working flow should be like this:

    - Requirement: A sample video, position
      1. mean_frame = get_meanframe(video_name)
         equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame)) 
      2. pd = PolygonDrawer(equ_meanframe,ori_positions, maxHeight = 960, maxWidth = 960)
         warped_image, warped_positions, M  = pd.run()
'''

import numpy as np
import cv2

def get_meanframe(video_name:str):
    '''
    Parameter
    ---------
    video_name: str, required
        The directory of a sample video.
    '''
    cap = cv2.VideoCapture(video_name)
    length = np.int64(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(length):    # Capture frame-by-frame
        ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
        if i == 0: # initialize mean frame
            mean_frame = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # Our operations on the frame come here    
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/length
        # img = frame/length
        mean_frame = mean_frame + img
    
    return mean_frame

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts, maxHeight = 960, maxWidth = 960):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# maxHeight = max(int(heightA), int(heightB))
	
    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# warped_positions = cv2.perspectiveTransform(np.array([ori_positions]) , M)
	# return the warped image
	return warped_image, M

FINAL_LINE_COLOR = (255, 100, 0)
WORKING_LINE_COLOR = (127, 127, 127) 

class PolygonDrawer(object):
    def __init__(self, equ_meanframe, ori_positions, maxHeight = 960, maxWidth = 960):
        self.window_name = "Original: select 4 maze corners" # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.equ_meanframe = equ_meanframe
        self.ori_positions = ori_positions
        self.maxHeight = maxHeight
        self.maxWidth = maxWidth


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every smouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("    Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("    Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.equ_meanframe)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self.equ_meanframe
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            cv2.polylines(canvas,  np.int32([self.ori_positions]), False, FINAL_LINE_COLOR, 1)

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                # And  also show what the current segment would look like
                # cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = self.equ_meanframe

        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]),True, FINAL_LINE_COLOR, thickness = 5)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        
        # Four points transform
        warped_image, M = four_point_transform(self.equ_meanframe, np.asarray(self.points), maxHeight = self.maxHeight, maxWidth = self.maxWidth)
        cv2.imshow("Processed Maze", warped_image)
        warped_positions = cv2.perspectiveTransform(np.array([self.ori_positions]) , M)[0]
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("Processed Maze")
       
        return warped_image, warped_positions, M

