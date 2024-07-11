#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo = YOLO("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt")  # YOLO v8 모델 경로 지정
        self.yolo.to('cpu')  # 모델을 CPU로 전환
        self.image_sub = rospy.Subscriber("/camera_nodelet2/image_out/compressed", CompressedImage, self.image_callback)  # 카메라 이미지 토픽 구독
        self.result_pub = rospy.Publisher("/yolo_results", String, queue_size=1)  # 결과 문자열 토픽 발행
        self.frame_count = 0  # 프레임 카운터 초기화

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return

        try:
            # ROS 압축 이미지 메시지를 OpenCV 이미지로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 컬러 이미지로 변환

            # YOLO v8을 사용하여 객체 감지 수행
            results = self.yolo(cv_image)

            # 객체 감지 결과 처리
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    obj_name = result.names[int(box.cls)]
                    confidence = box.conf.item()
                    detected_objects.append(f"{obj_name}: {confidence:.2f}")

            # 결과를 문자열로 변환하여 발행
            if detected_objects:
                result_str = "; ".join(detected_objects)
            else:
                result_str = "No detections"
            self.result_pub.publish(result_str)
            rospy.loginfo(f"Published YOLO results: {result_str}")

        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))


if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()
