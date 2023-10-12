# include"img_seg/seg_node.h"

image_transport::Publisher img_pub;
cv_bridge::CvImagePtr cv_ptr = boost::make_shared<cv_bridge::CvImage>();
sensor_msgs::Image image_msg;

void process_res(cv::Mat &input);
cv::Point2d pix2cam(const int &x, const int& y);
std::string coordinate_info(const cv::Point2d &coordinate);
double caculate_angel(const std::vector<cv::Point> &weld_contours);

std::vector<double> projection_matrix{1419.6726, 1419.9984, 671.2795, 530.6760};
double scale = 0.277; //distance between ground and camera, unit meter

bool pub_msg = true;
double count = 0;


void img_callback(sensor_msgs::ImageConstPtr img_msg, SampleSegmentation* seg_system){
    cv::Mat img = rosIMG2mat(img_msg);   
    cv::Mat result = seg_system->infer(img, 256, 256);   
    // cv::Mat result = seg_system->infer(img, 320, 256);
    cv::resize(result, result, cv::Size(1280, 1024));
    process_res(result);
    cv_ptr->image = result;
    image_msg = *(cv_ptr->toImageMsg());
    image_msg.header = img_msg->header;
    img_pub.publish(image_msg);
    
    // count += 0.1;
    // write_value(*route, "MAIN.FL_ActVelo", count);
}

// void img_callback(sensor_msgs::ImageConstPtr img_msg, SampleSegmentation* seg_system,  AdsDevice* route){
//     cv::Mat img = rosIMG2mat(img_msg);      
//     cv::Mat result = seg_system->infer(img, 320, 256);
//     cv::resize(result, result, cv::Size(1280, 1024));
//     process_res(result);
//     cv_ptr->image = result;
//     image_msg = *(cv_ptr->toImageMsg());
//     image_msg.header = img_msg->header;
//     img_pub.publish(image_msg);
    
//     // count += 0.1;
//     // write_value(*route, "MAIN.FL_ActVelo", count);
// }

int main(int argc, char **argv){
    ros::init(argc, argv, "seg_node");
    ros::NodeHandle n;
    ROS_WARN("---SYSTEM INITIALIZING---");

    ROS_WARN("CONSTRUCTING ADS ROUTE.....");
    static const AmsNetId remoteNetId{192, 168, 1, 2, 1, 1};
    static const char remoteIpV4[] = "192.168.1.2";
    // AdsDevice route{remoteIpV4, remoteNetId, 851};


    ROS_WARN("DESERIALIZING ENGINE .....");
    SampleSegmentation seg_system("/home/nvidia/data_1t/sds_ws/ros_ws/src/img_seg/onnx_module/models0901.engine");

     ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/hikrobot_camera/rgb", 15, boost::bind(img_callback, _1, &seg_system));
    // ros::Subscriber img_sub = n.subscribe<sensor_msgs::Image>("/hikrobot_camera/rgb", 15, boost::bind(img_callback, _1, &seg_system, &route));
    image_transport::ImageTransport it(n);
    img_pub = it.advertise("/predicted_img", 1000);
    cv_ptr->encoding = sensor_msgs::image_encodings::MONO8; // 就是rgb格式

    ROS_WARN("ALL DOWN! SYSTEM START!");

    ros::spin();
  
    return 0;
}

cv::Mat rosIMG2mat(sensor_msgs::ImageConstPtr img_msg){
  cv::Mat img;
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img_msg, "rgb8");
  cv_ptr->image.copyTo(img);

  return img;
}

void process_res(cv::Mat &input){
  std::vector<cv::Vec4i> hierachy;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(input,  contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  if (contours.size()){
      int max_index = 0;

      // filter wrong segmentation
      for (size_t i = 0; i < contours.size(); i++)
      {
        if(cv::contourArea(contours[i]) > cv::contourArea(contours[max_index])){
          max_index = i;
        }
      }
      std::cout << "pixel number of weld: " << contours[max_index].size() << std::endl;

      // caculate center
      int x = 0;  int y = 0;

    cv::RotatedRect r_rect = cv::minAreaRect(contours[max_index]);
    x = r_rect.center.x;
    y = r_rect.center.y;
    cv::Point2d cam_coordinate = pix2cam(x, y);
    double degree = caculate_angel(contours[max_index]);
    cv::drawMarker(input, cv::Point(x,y), cv::Scalar(0,0,0),cv::MARKER_CROSS, 10, 3);
    cv::putText(input, "The CAM Coordinate" , cv::Point(800,750), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
    cv::putText(input, coordinate_info(cam_coordinate), cv::Point(800, 800), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        cv::putText(input, "The Weld Gradient" , cv::Point(800,850), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
    cv::putText(input, std::to_string(degree), cv::Point(800, 900), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
  }
  
  else{
    std::cout << "no countours" << std::endl;
  }
  
}

// unit mm
cv::Point2d pix2cam(const int &x, const int& y){
  cv::Point2d tmp;
  tmp.x = (x - projection_matrix[2]) / projection_matrix[0] * scale * 1000;
  tmp.y = (y - projection_matrix[3]) / projection_matrix[1] * scale * 1000;
  return tmp;
}

double caculate_angel(const std::vector<cv::Point> &weld_contours){
  cv::RotatedRect r_rect = cv::minAreaRect(weld_contours);

  //find vertex
  int top_index = 0; int bottom_index = 0; int left_index = 0; int right_index = 0;
  cv::Point2f vtx[4];
  r_rect.points(vtx);
  for(int i = 0; i < 4; i++){
     if(vtx[i].y > vtx[top_index].y){
        top_index = i;
     }
      if(vtx[i].y < vtx[bottom_index].y){
        bottom_index = i;
     }
      if(vtx[i].x < vtx[left_index].x){
        left_index = i;
     }
      if(vtx[i].x > vtx[right_index].x){
        right_index = i;
     }
  }

  //find longer edge and return degree
  double degree;
  if(pow(vtx[top_index].x - vtx[left_index].x, 2) + pow(vtx[top_index].y - vtx[left_index].y, 2)  > 
       pow(vtx[top_index].x - vtx[right_index].x, 2) + pow(vtx[top_index].y - vtx[right_index].y, 2)){
        double k = -(vtx[top_index].y - vtx[left_index].y) / (vtx[top_index].x - vtx[left_index].x);
        degree = atan(k);
       }
  else{
      double k = -(vtx[top_index].y - vtx[right_index].y) / (vtx[top_index].x - vtx[right_index].x);
      degree = atan(k);
  }
  degree = degree / 3.1416926 * 180;
  return degree;
}

std::string coordinate_info(const cv::Point2d &coordinate){
    std::stringstream ss;
    ss.precision(2);
    ss.setf(std::ios::fixed);
    ss << "[" << coordinate.x  << " , " << coordinate.y << "]" ;
    return ss.str(); 
}

