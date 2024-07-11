#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import onnxruntime as ort
import numpy as np
import concurrent.futures

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.session = ort.InferenceSession('/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.onnx')
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
            input_image = self.preprocess(cv_image_bgr)

            # ONNX 모델에 입력
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_image})

            # 결과 처리 및 객체 감지 정보 추가
            self.postprocess_and_publish(cv_image_bgr, outputs)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def preprocess(self, image):
        # 이미지 전처리: 크기 조정 및 차원 변경
        image = cv2.resize(image, (640, 640))
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image.astype(np.float32) / 255.0  # Normalize
        return image

    def postprocess_and_publish(self, image, outputs):
        # ONNX 모델의 출력 처리 및 객체 감지 결과 추가
        # (이 부분은 YOLO v8의 출력 형식에 맞게 조정해야 합니다)
        detections = outputs[0]

        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection[:6]
            if confidence >= 0.7:  # 정확도가 70% 이상인 경우에만 표시
                # 바운딩 박스 그리기
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 객체명과 정확도 텍스트 추가
                label = f"Object: {confidence:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 이미지를 압축하여 ROS 메시지로 변환
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # 압축 품질 설정 (0-100, 높을수록 품질이 높음)
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = rospy.Time.now()
        compressed_msg.format = "jpeg"
        compressed_msg.data = cv2.imencode('.jpg', image, compression_params)[1].tobytes()

        # 압축된 이미지 메시지를 발행
        self.image_pub.publish(compressed_msg)

if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()
