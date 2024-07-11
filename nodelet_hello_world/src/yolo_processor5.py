#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import concurrent.futures

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo = YOLO("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt")  # YOLO v8 모델 경로 지정
        self.yolo.to('cpu')  # 모델을 CPU로 전환
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)  # 카메라 이미지 토픽 구독
        self.image_pub = rospy.Publisher("/yolo_output/compressed", CompressedImage, queue_size=1)  # 압축 이미지 토픽 발행
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # 스레드 풀 생성

    def image_callback(self, msg):
        self.executor.submit(self.process_image, msg)

    def process_image(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환 (흑백 이미지 처리)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")

            # YOLO 모델이 BGR 이미지를 기대하기 때문에 흑백 이미지를 BGR로 변환
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

            # YOLO v8을 사용하여 객체 감지 수행
            results = self.yolo(cv_image_bgr)

            # 감지된 객체 정보 이미지에 추가
            for result in results:
                for box in result.boxes:
                    confidence = box.conf.item()
                    if confidence >= 0.7:  # 정확도가 70% 이상인 경우에만 표시
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 클래스 이름
                        obj_name = result.names[int(box.cls)]

                        # 바운딩 박스 그리기
                        cv2.rectangle(cv_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 객체명과 정확도 텍스트 추가
                        label = f"{obj_name}: {confidence:.2f}"
                        cv2.putText(cv_image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 이미지를 압축하여 ROS 메시지로 변환
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # 압축 품질 설정 (0-100, 높을수록 품질이 높음)
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.format = "jpeg"
            compressed_msg.data = cv2.imencode('.jpg', cv_image_bgr, compression_params)[1].tobytes()

            # 압축된 이미지 메시지를 발행
            self.image_pub.publish(compressed_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()
