#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import pymysql
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo = YOLO("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt")  # YOLO v8 모델 경로 지정
        self.yolo.to('cpu')  # 모델을 CPU로 전환
        self.image_sub = rospy.Subscriber("/camera_nodelet2/image_out/compressed", CompressedImage, self.image_callback)  # 카메라 이미지 토픽 구독
        self.result_pub = rospy.Publisher("/yolo_results", String, queue_size=1)  # 결과 문자열 토픽 발행
        self.frame_count = 0  # 프레임 카운터 초기화
        self.confidence_threshold = 0.5  # 저장할 객체의 최소 신뢰도 설정

        # 데이터베이스 연결 설정
        try:
            self.db = pymysql.connect(
                host="13.124.83.151",
                user="root",
                password="1235",
                database="rpm"
            )
        except pymysql.MySQLError as e:
            rospy.logerr(f"Error connecting to the database: {e}")
            self.db = None

    def save_to_database(self, detected_objects):
        if self.db is None:
            rospy.logerr("Database connection is not available.")
            return

        cursor = self.db.cursor()

        for obj in detected_objects:
            class_name = obj['name']
            confidence = obj['confidence']

            # SQL 쿼리 작성 및 실행
            sql = """
            INSERT INTO detections (detections_no, class_name, confidence)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE class_name = VALUES(class_name), confidence = VALUES(confidence)
            """
            cursor.execute(sql, ('1', class_name, confidence))  # '1' 키를 사용하여 하나의 행만 삽입 또는 업데이트

        self.db.commit()
        cursor.close()

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
                    if confidence >= self.confidence_threshold:
                        detected_objects.append({
                            'name': obj_name,
                            'confidence': confidence
                        })

            # 결과를 문자열로 변환하여 발행
            if detected_objects:
                result_str = "; ".join([f"{obj['name']}: {obj['confidence']:.2f}" for obj in detected_objects])
            else:
                result_str = "No detections"
            self.result_pub.publish(result_str)
            rospy.loginfo(f"Published YOLO results: {result_str}")

            # 데이터베이스에 저장
            self.save_to_database(detected_objects)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    rospy.init_node('yolo_processor', anonymous=True)
    yolo_processor = YOLOProcessor()
    rospy.spin()

    # 데이터베이스 연결 해제
    if yolo_processor.db is not None:
        yolo_processor.db.close()
