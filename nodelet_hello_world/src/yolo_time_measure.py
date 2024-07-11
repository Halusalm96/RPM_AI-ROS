#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import time
import requests
import websocket
import threading

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo = YOLO("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt")  # YOLO v8 모델 경로 지정
        self.yolo.to('cpu')  # 모델을 CPU로 전환
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)  # 카메라 이미지 토픽 구독
        self.image_pub = rospy.Publisher("/yolo_output/compressed", CompressedImage, queue_size=1)  # 압축 이미지 토픽 발행

    def image_callback(self, msg):
        try:
            # 시작 시간 측정
            start_time = time.time()

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

            # 웹 서버로 이미지 전송 (HTTP POST)
            post_time = self.send_image_via_post(compressed_msg.data)

            # 웹 서버로 이미지 전송 (WebSocket)
            websocket_time = self.send_image_via_websocket(compressed_msg.data)

            # 끝 시간 측정
            end_time = time.time()

            # 시간 출력
            rospy.loginfo(f"Processing Time: {end_time - start_time:.4f} seconds")
            rospy.loginfo(f"HTTP POST Transmission Time: {post_time:.4f} seconds")
            rospy.loginfo(f"WebSocket Transmission Time: {websocket_time:.4f} seconds")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def send_image_via_post(self, compressed_data):
        response_time_start = time.time()
        response = requests.post("http://rpm-web.p-e.kr/", files={"file": compressed_data})
        response_time_end = time.time()

        if response.status_code == 200:
            rospy.loginfo("Image successfully sent to web server.")
        else:
            rospy.logerr("Failed to send image to web server.")

        return response_time_end - response_time_start

    def send_image_via_websocket(self, compressed_data):
        response_time_start = time.time()

        def on_message(ws, message):
            rospy.loginfo("Received response from server")
            ws.close()

        def on_error(ws, error):
            rospy.logerr(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            rospy.loginfo("WebSocket connection closed")

        ws = websocket.WebSocketApp("ws://http://rpm-web.p-e.kr/",
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)

        ws.on_open = lambda ws: ws.send(compressed_data)

        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        wst.join()

        response_time_end = time.time()

        return response_time_end - response_time_start


if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()
