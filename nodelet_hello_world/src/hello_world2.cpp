#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace nodelet_hello_world
{

class Hello2 : public nodelet::Nodelet
{
private:
    virtual void onInit()
    {
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        NODELET_DEBUG("Hello2 노드렛 초기화 완료");

        // 퍼블리셔와 서브스크라이버 초기화
        pub = private_nh.advertise<sensor_msgs::CompressedImage>("image_out/compressed", 5);
        sub = private_nh.subscribe("image_in", 5, &Hello2::callback, this);
    }

    void callback(const sensor_msgs::ImageConstPtr& input)
    {
        try
        {
            // ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::BGR8);

            // 이미지를 320x240 크기로 리사이즈
            cv::Mat resized_image;
            cv::resize(cv_ptr->image, resized_image, cv::Size(320, 240));

            // 이미지를 그레이스케일로 변환
            cv::Mat gray_image;
            cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);

            // OpenCV 이미지를 다시 ROS 이미지 메시지로 변환하고 압축
            std_msgs::Header header = input->header; // 입력 이미지와 동일한 타임스탬프와 tf 프레임
            sensor_msgs::CompressedImage compressed_msg;
            compressed_msg.header = header;
            compressed_msg.format = "jpeg";

            // JPEG 압축 품질 설정 (0-100, 높을수록 품질이 높음)
            std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};
            cv::imencode(".jpg", gray_image, compressed_msg.data, compression_params);

            // 압축된 이미지 발행
            pub.publish(compressed_msg);
        }
        catch (cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge 예외: %s", e.what());
        }
    }

    ros::Publisher pub;
    ros::Subscriber sub;
};

}

PLUGINLIB_EXPORT_CLASS(nodelet_hello_world::Hello2, nodelet::Nodelet);
