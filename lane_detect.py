import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from numpy import linalg as la


class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        # self.pub_box = rospy.Publisher("lane_detection/box", Image, queue_size=1)
        self.pub_waypoint = rospy.Publisher("lane_detection/waypoint", Int32MultiArray, queue_size=1)
        self.steering_angle = rospy.Publisher("lane_detection/steering_angle", Float32, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image, waypoint, steering_angle = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
            # out_box_msg = self.bridge.cv2_to_imgmsg(box_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)
            # self.pub_box.publish(out_box_msg)
            waypoint_data = Int32MultiArray()
            waypoint_data.data = waypoint
            self.pub_waypoint.publish(waypoint_data)
            self.steering_angle.publish(steering_angle)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO

        ####
        blurImg = cv2.GaussianBlur(img, (5, 5), 7)
        edged = cv2.Canny(blurImg, 100, 200)
        return edged

    # 90, 255 FOR GAZEBO
    # 250, 255 FOR ROSBAG
    def color_thresh(self, img, thresh=(160, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO

        ####
        imgHSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
        imgHSL[:, :, 0] = np.where(imgHSL[:, :, 0] < thresh[0], 0, 1)
        # imgHSL[:, :, 1] = np.where(imgHSL[:, :, 1] < thresh[1], 0, 1)
        imgHSL[:, :, 1] = np.where(imgHSL[:, :, 1] > thresh[0], 1, 0)

        # # ROSBAG ADDED 
        imgHSL[:, :, 2] = np.where(imgHSL[:, :, 2] < thresh[0], 0, 1)  

        binary_output = cv2.cvtColor(imgHSL, cv2.COLOR_HLS2RGB)
        binary_output = cv2.cvtColor(binary_output, cv2.COLOR_RGB2GRAY)

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        ####

        SobelOutput =  self.gradient_thresh(img, np.inf, -np.inf)
        binaryImage = np.zeros_like(SobelOutput)
        ColorOutput = self.color_thresh(img)
        binaryImage[(ColorOutput==1)|(SobelOutput==255)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2).astype(np.uint8)

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO

        ####
        img = img.astype(np.uint8)

        original_points = np.float32([[370, 485], [890, 485], [1080, 715], [75, 715]])
        transformed_points = np.float32([[0, 0], [1280, 0], [1180, 720], [100, 720]])

        M = cv2.getPerspectiveTransform(original_points, transformed_points)
        Minv = np.linalg.inv(M)
        warped_img = cv2.warpPerspective(img, M, (1280, 720))

        # warped_img = cv2.warpPerspective(img, M, (width, height))

        return warped_img, M, Minv

    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)
        # box_image = None

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
            # box_image = ret['out_img']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']
                    # box_image = ret['out_img']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            #Generate midpoints to act as waypoints
            waypoint = None
            y = 180
            y_ = 360

            # left_midpoint = np.array([left_fit[0], left_fit[1], left_fit[2]-y])
            # left_x = np.roots(left_midpoint)
            # left_x = left_x[np.isreal(left_x)].real
            left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
            left_x_ = left_fit[0]*y_**2 + left_fit[1]*y_ + left_fit[2]

            # right_midpoint = np.array([right_fit[0], right_fit[1], right_fit[2]-y])
            # right_x = np.roots(right_midpoint)
            # right_x = right_x[np.isreal(right_x)].real
            right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            right_x_ = right_fit[0]*y_**2 + right_fit[1]*y_ + right_fit[2]

            x = (left_x + right_x)/2
            x_ = (left_x_ + right_x_)/2
            waypoint = [int(x), int(y)]
            x_ = int(x_)
            # waypoint_test = (480, 360)

            cv2.circle(bird_fit_img, (waypoint[0], waypoint[1]), 10, (255, 255, 255), -1)
            cv2.circle(bird_fit_img, (x_, y_), 10, (255, 255, 255), -1)

            cv2.line(bird_fit_img, (waypoint[0], waypoint[1]), (x_, y_), (255,0,0))
            cv2.line(bird_fit_img, (640,720), (640, 0), (0,0,255))

            v1 = np.array([640-waypoint[0], 720-waypoint[1]])
            # v1 = np.array([x_-waypoint[0], y_-waypoint[1]])
            v2 = np.array([0, 720])
            steering_angle = self.find_angle(v1, v2)
            if waypoint[0] > 640:
                steering_angle = -steering_angle

            # if np.abs(steering_angle) < 0.2:
            #     steering_angle = 0
            steering_angle = np.clip(steering_angle, -0.3, 0.3)

            return combine_fit_img, bird_fit_img, waypoint, steering_angle

if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)