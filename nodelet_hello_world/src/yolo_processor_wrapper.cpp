#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <boost/python.hpp>
#include <ros/ros.h>

namespace nodelet_hello_world
{

class YOLOProcessorWrapper : public nodelet::Nodelet
{
public:
    YOLOProcessorWrapper() {}
    ~YOLOProcessorWrapper() {}

private:
    virtual void onInit()
    {
        try
        {
            namespace py = boost::python;
            Py_Initialize();

            // Add the virtual environment to the Python path
            py::object sys = py::import("sys");
            sys.attr("path").attr("insert")(0, "/home/user1/catkin_ws/py3env/lib/python3.8/site-packages");
            
            // Add the script directory to the Python path
            sys.attr("path").attr("append")("/home/user1/catkin_ws/src/nodelet_hello_world/src");

            // Initialize sys.argv
            py::list argv;
            argv.append("");
            sys.attr("argv") = argv;
            
            // Import the main script
            yolo_processor_nodelet_ = py::import("yolo_processor_nodelet");
            
            // Call the main function in the script
            yolo_processor_nodelet_.attr("main")();
        }
        catch (boost::python::error_already_set&)
        {
            PyErr_Print();
            NODELET_ERROR("Failed to initialize YOLOProcessorNodelet");
        }
    }

    boost::python::object yolo_processor_nodelet_;
};

} // namespace nodelet_hello_world

PLUGINLIB_EXPORT_CLASS(nodelet_hello_world::YOLOProcessorWrapper, nodelet::Nodelet)
