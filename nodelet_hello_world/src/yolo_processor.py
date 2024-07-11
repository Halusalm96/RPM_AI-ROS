#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo = YOLO("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt")  # YOLO v8 모델 경로 지정
        self.yolo.to('cpu')  # 모델을 CPU로 전환
        # self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)  # 카메라 이미지 토픽 구독
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)  # 카메라 이미지 토픽 구독
        self.image_pub = rospy.Publisher("/yolo_output", Image, queue_size=1)  # 결과 이미지 토픽 발행

    def image_callback(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # YOLO v8을 사용하여 객체 감지 수행
            results = self.yolo(cv_image)

            # 감지된 객체 정보 이미지에 추가
            for result in results:
                for box in result.boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 클래스 이름
                    obj_name = result.names[int(box.cls)]
                    # 정확도
                    confidence = box.conf.item()

                    # 바운딩 박스 그리기
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 객체명과 정확도 텍스트 추가
                    label = f"{obj_name}: {confidence:.2f}"
                    cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 결과 이미지를 가져와서 ROS 이미지 메시지로 변환하여 발행
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))


if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()
