import cv2
import rospy
import torch
import numpy as np
from PIL import Image


from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from std_msgs.msg import Bool

class human_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb_raw/image_raw_color', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("human_detector/annotate", Image, queue_size=1)
        self.pub_detected = rospy.Publisher("/human_detector/detected", Bool, queue_size=1)

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, detected = self.object_detection(raw_img)

        if mask_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_detected.publish(detected)

    def object_detection(self, raw_img):
        """
        Get bounded objects from input image
        """
        device = 'cuda'

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # For the small model
        model = model.to(device)

        img = np.asarray(raw_img)

        results = model(img)

        confidence_threshold = 0.8
        human_boxes = []
        human_confidence = []

        detected = False

        for box in results.xyxy[0]:
            if box[5] == 0 and box[4] > confidence_threshold:
                human_boxes.append(box[:4])
                human_confidence.append(box[4])

                top_left = (int(box[0].item()), int(box[1].item()))
                bottom_right = (int(box[2].item()), int(box[3].item()))
                color = (255, 0, 0)
                thickness = 5
                cv2.rectangle(raw_img, top_left, bottom_right, color, thickness)

                X_center = (box[0].item() + box[2].item())/2
                Y_bottom = box[3]

                if X_center > 350 and X_center < 935:
                    if Y_bottom > 350 and Y_bottom < 720:
                        detected = True

        # img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        box_img = np.array(raw_img)

        return box_img, detected


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    human_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)