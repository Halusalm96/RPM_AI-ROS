#include <ros/ros.h>
#include <nodelet/loader.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolo_processor_manager");
    nodelet::Loader manager(false);  // false to disable the manager's automatic unloading
    nodelet::M_string remappings(ros::names::getRemappings());
    nodelet::V_string my_argv;
    manager.load(ros::this_node::getName(), "nodelet_hello_world/YOLOProcessorNodelet", remappings, my_argv);
    ros::spin();
    return 0;
}
