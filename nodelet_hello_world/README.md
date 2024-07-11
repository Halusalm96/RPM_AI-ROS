2024/5/29 urdf변형한 것 urdf 디렉터리에 존재 

nodelet 의존성 패키지 - pi
sudo apt-get install ros-melodic-usb-cam
sudo apt-get install pkg-config
sudo apt-get install libusb-1.0-0-dev
pip install rospkg
+opencv(2024/6/24에 간단한 pi 설치 내용 있음)

ONNX사용 패키지 - pc
pip install onnx onnxruntime
+ pytorch도 설치 필요
+opencv
echo 'export PYTHONPATH=$PYTHONPATH:/home/user1/.local/lib/python3.8/site-packages' >> ~/.bashrc source

C++용 - pc
sudo apt-get install libopencv-dev
sudo apt-get install onnxruntime


2024/6/24 12:00 nodelet 업로드 hello_world.launch를 라즈베리파이에서 실행
2024/6/24 12:40 launch/reciever.launch 업로드 -> 수신 받는 컴퓨터에서 실행
CMAKE를 하기 위해 CMakeLists.txt의 INCLUDE_DIRS include를 주석 처리 함 추후 카메라 문제 발생하면 봐야함

2024/6/24 16:45 src/yolo_processor.py, launch/yolo_processor.launch 생성 + PC의 CMakeLists.txt와 package.xml에 python, opencv, ultralytics 의존성 추가

2024/6/25 10:15 python-catkin-tools, python3-dev, python3-numpy 설치 -> 안되면 cv_bridge 직접 설치

주의 사항 : ros_master_uri, ros_ip 설정 필요(~/.bashrc), 사양 문제로 gpu가 아닌 cpu로 인식
입력 토픽 : /usb_cam/image_raw, 출력 토픽 : /yolo_output

2024/6/25 12:30 yolo_processor2.py를 통해 노드로 영상이 아닌 객체와 정확도만 송출, yolo_processor.launch 파일 수정(type만 변경하면 변환 가능)
입력 토픽 : /usb_cam/image_raw, 출력 토픽 : /yolo_results

2024/6/25 15:10 yolo_processor에서 영역, 객체명 보이게 수정
yolo_processor3를 통해 웹서버에서 인식 할 수 있는 압축 이미지로 변환 + 정확도 70% 이상일 때 바운딩 박스 생성

2024/6/25 15:30 hello_world2.cpp를 생성하여 이미지를 resize+흑백을 적용하여 YOLO의 속도를 향상하려고 함 + hello_world2.launch 수정

2024/6/25 17:00 yolo_processor4를 생성하여 hello_world2.cpp의 이미지 처리 -> 속도의 큰 차이가 없어서 원본 이미지 사용이 나아보임

2024/6/26 10:20 yolo_processor4에 이미지 압축 품질 설정 코드 삽입 - 추가 압축 가능
hello_world2도 이미지 압출 품질 설정 코드 삽임

pi_yolo_test를 통해 pi에서 yolo 시도 -> pi 사양을 파악해 보니 불가할 것으로 판단됨

2024/6/18 14:40 yolo_processor5를 통해 기존 이미지 압축, 흑백변환에 더불어 멀티 스레딩 적용
yolo_processor6를 통해 YOLO 모델에 ONNX 적용 - 비교 필요

2024/7/1 11:30 yolo_processor.cpp을 통해 nodelet을 통하여 영상 송출 -> cpp헤더가 없어 python으로 바인딩
noelet_plugins.xml, src/tolo_processor.py, src/yolo_processor_manager.cpp, src/yolo_processor_wrapper.cpp, launch/tolo_processor_manager.launch 생성


2024/7/8 12:10 최종 - src/yolo_db.py, hello_world2.cpp사용
pi - roslaunch yolo_processor_nodelet hello_world2.launch
pc - roslaunch nodelet_hello_world yolo_db.launch로 실행