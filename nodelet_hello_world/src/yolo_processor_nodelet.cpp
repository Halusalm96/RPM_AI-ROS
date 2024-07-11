#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ultralytics/YOLO.hpp>

namespace nodelet_hello_world
{

class YOLOProcessorNodelet : public nodelet::Nodelet
{
public:
    YOLOProcessorNodelet() = default;
    ~YOLOProcessorNodelet() = default;

private:
    virtual void onInit()
    {
        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        image_sub_ = nh.subscribe("/usb_cam/image_raw", 1, &YOLOProcessorNodelet::imageCallback, this);
        image_pub_ = nh.advertise<sensor_msgs::CompressedImage>("/yolo_output/compressed", 1);
        yolo_ = std::make_shared<ultralytics::YOLO>("/home/user1/catkin_ws/src/nodelet_hello_world/yolov8s.pt");
        yolo_->to("cpu");
        bridge_ = std::make_shared<cv_bridge::CvBridge>();
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            cv::Mat cv_image = bridge_->imgmsg_to_cv2(msg, "mono8");
            cv::Mat cv_image_bgr;
            cv::cvtColor(cv_image, cv_image_bgr, cv::COLOR_GRAY2BGR);

            auto results = yolo_->detect(cv_image_bgr);

            for (const auto& result : results)
            {
                for (const auto& box : result.boxes)
                {
                    float confidence = box.conf;
                    if (confidence >= 0.7)
                    {
                        cv::rectangle(cv_image_bgr, cv::Rect(cv::Point(box.xyxy[0], box.xyxy[1]), cv::Point(box.xyxy[2], box.xyxy[3])), cv::Scalar(0, 255, 0), 2);
                        cv::putText(cv_image_bgr, result.names[box.cls] + ": " + std::to_string(confidence), cv::Point(box.xyxy[0], box.xyxy[1] - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }

            std::vector<uchar> buf;
            cv::imencode(".jpg", cv_image_bgr, buf, {cv::IMWRITE_JPEG_QUALITY, 90});
            sensor_msgs::CompressedImage compressed_msg;
            compressed_msg.header.stamp = ros::Time::now();
            compressed_msg.format = "jpeg";
            compressed_msg.data = buf;

            image_pub_.publish(compressed_msg);
        }
        catch (const cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
        }
        catch (const std::exception& e)
        {
            NODELET_ERROR("Exception: %s", e.what());
        }
    }

    ros::Subscriber image_sub_;
    ros::Publisher image_pub_;
    std::shared_ptr<cv_bridge::CvBridge> bridge_;
    std::shared_ptr<ultralytics::YOLO> yolo_;
};

} // namespace nodelet_hello_world

PLUGINLIB_EXPORT_CLASS(nodelet_hello_world::YOLOProcessorNodelet, nodelet::Nodelet)
