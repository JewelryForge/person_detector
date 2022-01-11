#include <chrono>
#include <iostream>

#include "calibrator.h"
#include "common.hpp"
#include "cuda_utils.h"
#include "logging.h"
#include "preprocess.h"
#include "utils.h"

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH \
  3000 * 3000  // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE =
    Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) +
        1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT
// boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

void doInference(IExecutionContext &context, cudaStream_t &stream,
                 void **buffers, float *output, int batchSize) {
  // infer on the batch asynchronously, and DMA output back to host
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, buffers[1],
                             batchSize * OUTPUT_SIZE * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char **argv, std::string &engine, std::string &img_dir) {
  if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
    return true;
  } else {
    return false;
  }
}

class Detector {
 public:
  explicit Detector(const std::string &engine_name, std::vector<int> classes): classes(std::move(classes)) {
    std::ifstream file(engine_name, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    auto *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) &buffers[inputIndex],
                          BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &buffers[outputIndex],
                          BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&stream));
    // prepare input data cache in pinned memory
    CUDA_CHECK(cudaMallocHost((void **) &img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void **) &img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  }

  void detect(cv::Mat &img, std::vector<cv::Rect> &res, bool annotate = true, bool timing = false) {
    assert((BATCH_SIZE == 1));
    float prob[BATCH_SIZE * OUTPUT_SIZE];
    auto *buffer_idx = (float *) buffers[inputIndex];
    assert(not img.empty());
    size_t size_image = img.cols * img.rows * 3;
    // copy data to pinned memory
    memcpy(img_host, img.data, size_image);
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image,
                               cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W,
                          INPUT_H, stream);
    // Run inference
    if (timing) {
      using namespace std::chrono;
      auto start = system_clock::now();
      doInference(*context, stream, (void **) buffers, prob, BATCH_SIZE);
      auto end = system_clock::now();
      std::cout << "inference time: " << duration_cast<milliseconds>(end - start).count()
                << "ms" << std::endl;
    } else {
      doInference(*context, stream, (void **) buffers, prob, BATCH_SIZE);
    }
    std::vector<Yolo::Detection> detections;
    nms(detections, prob, CONF_THRESH, NMS_THRESH, classes);
    for (auto &det: detections) {
      cv::Rect r = get_rect(img, det.bbox);
      res.push_back(r);
      if (annotate) {
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int) det.class_id),
                    cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
      }
    }
  }

  ~Detector() {
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
  }

 private:
  cudaStream_t stream{};
  uint8_t *img_host = nullptr;
  uint8_t *img_device = nullptr;
  float *buffers[2]{};
  int inputIndex, outputIndex;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;
  IRuntime *runtime = nullptr;
  Logger gLogger;
  std::vector<int> classes;
};

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
// #include "lidarmarkers_msg/lidarMarkers.h"
#include "detection_core.h"
#include "box.h"
#include <Eigen/Core>

cv::Mat projectVelo2imageKitti(const cv::Mat &xyz) {
  cv::Mat R_rect_00 = (cv::Mat_<double>(3, 3)
      <<
      9.998817e-01, 1.511453e-02, -2.841595e-03,
      -1.511724e-02, 9.998853e-01, -9.338510e-04,
      2.827154e-03, 9.766976e-04, 9.999955e-01);
  cv::Mat P_rect_02 = (cv::Mat_<double>(3, 4)
      <<
      7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
      0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
      0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03);

  cv::Mat R = (cv::Mat_<double>(3, 3)
      <<
      7.533745e-03, -9.999714e-01, -6.166020e-04,
      1.480249e-02, 7.280733e-04, -9.998902e-01,
      9.998621e-01, 7.523790e-03, 1.480755e-02);

  cv::Mat T = (cv::Mat_<double>(3, 1)
      << -4.069766e-03, -7.631618e-02, -2.717806e-01);

  cv::Mat X = R_rect_00 * (R * xyz + T);

  cv::Mat X_hom = (cv::Mat_<double>(4, 1)
      << X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0), 1);

  cv::Mat imgCoord = P_rect_02 * X_hom;
  double x = imgCoord.at<double>(0, 0), y = imgCoord.at<double>(1, 0), z = imgCoord.at<double>(2, 0);
//  ROS_INFO_STREAM("IMG: " << x / z << ' ' << y / z << ' ' << z);
  return (cv::Mat_<double>(3, 1) << x / z, y / z, z);
}

cv::Mat projectVelo2imageKaist(const cv::Mat &xyz) {
  cv::Mat P_left = (cv::Mat_<float>(3, 4)
      <<
      7.8135803071870055e+02, 0., 6.1796599578857422e+02, 0.,
      0., 7.8135803071870055e+02, 2.6354596138000488e+02, 0.,
      0., 0., 1., 0.);
  cv::Mat Vehicle2LeftVLP = (cv::Mat_<float>(4, 4)
      <<
      -0.516377, -0.702254, -0.490096, -0.334623,
      0.491997, -0.711704, 0.501414, 0.431973,
      -0.700923, 0.0177927, 0.713015, 1.94043,
      0, 0, 0, 1);
  cv::Mat Vehicle2Stereo = (cv::Mat_<float>(4, 4)
      <<
      -0.00680499, -0.0153215, 0.99985, 1.64239,
      -0.999977, 0.000334627, -0.00680066, 0.247401,
      -0.000230383, -0.999883, -0.0153234, 1.58411,
      0, 0, 0, 1);
  cv::Mat Tr = Vehicle2Stereo * Vehicle2LeftVLP.inv();
  cv::Mat P_veloToImg = P_left * Tr;
  return P_veloToImg * xyz;
}

class DetectorNode {
 public:
  explicit DetectorNode(ros::NodeHandle &nh,
                        const std::string &image_topic,
                        const std::string &velo_topic,
                        const std::string &engine_name) : detector(engine_name, {2}),
                                                          lidar_detector(nh),
                                                          pointcloud(new pcl::PointCloud<pcl::PointXYZ>) {
    image_sub = nh.subscribe<sensor_msgs::Image>(image_topic, 1, &DetectorNode::imageUpdate, this);
    velo_sub = nh.subscribe<sensor_msgs::PointCloud2>(velo_topic, 1, &DetectorNode::detectAndPublish, this);
    ROS_INFO_STREAM("Subscribing to " << image_topic << " and " << velo_topic);
    anno_pub = nh.advertise<sensor_msgs::Image>("/annotated", 1);
    bbox_pub = nh.advertise<visualization_msgs::MarkerArray>("markers", 1);
    bbox_raw_pub = nh.advertise<visualization_msgs::MarkerArray>("all_markers", 1);
  }

  void detectAndPublish(const sensor_msgs::PointCloud2::ConstPtr &msg_pc2) {
    if (image == nullptr) return;
    pcl::fromROSMsg(*msg_pc2, *pointcloud);
    if (pointcloud->empty()) return;
    std::vector<BoundingBoxCalculator::BoundingBox> boxes;
    lidar_detector.callback(msg_pc2, boxes);

    std::vector<cv::Rect> res;
    detector.detect(image->image, res, true);
    std::vector<cv::Mat> center_of_boxes_proj;
    for (const auto &box: boxes) {
      cv::Mat xyz = (cv::Mat_<double>(3, 1)
          << box.center.x, box.center.y, box.center.z);
      auto uvw = projectVelo2imageKitti(xyz);
      center_of_boxes_proj.push_back(uvw);
      if (uvw.at<double>(2, 0) > 0) {
        cv::circle(image->image,
                   cv::Point(uvw.at<double>(0, 0), uvw.at<double>(1, 0)),
                   10,
                   cv::Scalar(0, 0, 255),
                   3);
      }
    }
    std::vector<BoundingBoxCalculator::BoundingBox> useful_boxes;
    for (const auto &rect: res) {
      double u, v, w, minz = -1;
      size_t minz_idx;
      for (size_t i = 0; i < center_of_boxes_proj.size(); ++i) {
        auto uvw = center_of_boxes_proj[i];
        u = uvw.at<double>(0, 0);
        v = uvw.at<double>(1, 0);
        w = uvw.at<double>(2, 0);
          if (w > 0 && rect.contains({static_cast<int>(u), static_cast<int>(v)})) {
          ROS_INFO_STREAM(u << ' ' << v << ' ' << w << '\n'
                            << rect << '\n');
          if (minz == -1 || minz > w) {
            minz_idx = i;
            minz = w;
          }
        }
      }
      ROS_INFO_STREAM("\n\n");
      if (minz != -1) {
        useful_boxes.push_back(boxes[minz_idx]);
      } else {
        // TODO: FINISH
      }
    }
    bbox_pub.publish(lidar_detector.makeMarkerArray(useful_boxes, msg_pc2->header));
    bbox_raw_pub.publish(lidar_detector.makeMarkerArray(boxes, msg_pc2->header));
    anno_pub.publish(image->toImageMsg());
  }

  void imageUpdate(const sensor_msgs::Image::ConstPtr &image_msg) {
    image = cv_bridge::toCvCopy(image_msg, "bgr8");
  }
 private:
  Detector detector;
//  LidarDetector lidar_detector;
  DetectionCore lidar_detector;
  ros::Subscriber image_sub, velo_sub;
  ros::Publisher anno_pub, bbox_pub, bbox_raw_pub;
  cv_bridge::CvImagePtr image;
  cv::Mat P_velo2img;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud;
};

int main(int argc, char **argv) {
  cudaSetDevice(DEVICE);
  ros::init(argc, argv, "detector_node");
  ros::NodeHandle nh;
  std::string image_topic, velo_topic, engine_path;
  nh.getParam("image_topic", image_topic);
  nh.getParam("velo_topic", velo_topic);
  nh.getParam("engine_path", engine_path);
  auto detector = DetectorNode(nh, image_topic, velo_topic, engine_path);

  while (ros::ok()) {
    ros::spinOnce();
  }

  //  auto detector = Detector(engine_name);
//  for (const auto &file_name: file_names) {
//    cv::Mat img = cv::imread(img_dir + "/" + file_name);
//    std::vector<Yolo::Detection> res;
//    detector.detect(img, res, file_name);
//  }
  return 0;
}
