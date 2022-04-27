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
  explicit Detector(const std::string &engine_name, std::vector<int> classes) : classes(std::move(classes)) {
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
    for (auto &det : detections) {
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

#include <vector>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

//cv::Mat projectVelo2imageKitti(const cv::Mat &xyz) {
//  cv::Mat R_rect_00 = (cv::Mat_<double>(3, 3)
//      <<
//      9.998817e-01, 1.511453e-02, -2.841595e-03,
//      -1.511724e-02, 9.998853e-01, -9.338510e-04,
//      2.827154e-03, 9.766976e-04, 9.999955e-01);
//  cv::Mat P_rect_02 = (cv::Mat_<double>(3, 4)
//      <<
//      7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
//      0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
//      0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03);
//
//  cv::Mat R = (cv::Mat_<double>(3, 3)
//      <<
//      7.533745e-03, -9.999714e-01, -6.166020e-04,
//      1.480249e-02, 7.280733e-04, -9.998902e-01,
//      9.998621e-01, 7.523790e-03, 1.480755e-02);
//
//  cv::Mat T = (cv::Mat_<double>(3, 1)
//      << -4.069766e-03, -7.631618e-02, -2.717806e-01);
//
//  cv::Mat X = R_rect_00 * (R * xyz + T);
//
//  cv::Mat X_hom = (cv::Mat_<double>(4, 1)
//      << X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0), 1);
//
//  cv::Mat imgCoord = P_rect_02 * X_hom;
//  double x = imgCoord.at<double>(0, 0), y = imgCoord.at<double>(1, 0), z = imgCoord.at<double>(2, 0);
////  ROS_INFO_STREAM("IMG: " << x / z << ' ' << y / z << ' ' << z);
//  return (cv::Mat_<double>(3, 1) << x / z, y / z, z);
//}
//
//cv::Mat projectVelo2imageCustom(const cv::Mat &xyz) {
//  cv::Mat X_hom = (cv::Mat_<double>(4, 1)
//      << xyz.at<double>(0, 0), xyz.at<double>(1, 0), xyz.at<double>(2, 0), 1);
//  cv::Mat P_cam = (cv::Mat_<double>(3, 4)
//      <<
//      256, 0., 256, 0.,
//      0., 256, 256, 0.,
//      0., 0., 1., 0.);
//  cv::Mat vehicle2velo = (cv::Mat_<double>(4, 4)
//      << 1., 0., 0., -1.13,
//      0., 1., 0., -0.08,
//      0., 0., 1., -1.86,
//      0, 0, 0, 1);
//  cv::Mat vehicle2cam = (cv::Mat_<double>(4, 4)
//      << 1., 0., 0., 0.39,
//      0., 1., 0., 0.0,
//      0., 0., 1., 1.2,
//      0, 0, 0, 1);
//
//  cv::Mat imgCoord = P_cam * vehicle2cam * vehicle2velo.inv() * X_hom;
//  double x = imgCoord.at<double>(0, 0), y = imgCoord.at<double>(1, 0), z = imgCoord.at<double>(2, 0);
//
//  return (cv::Mat_<double>(3, 1) << x / z, y / z, z);
//}

class ProjectionModelCustom {
 public:
  ProjectionModelCustom() {
    Eigen::Affine3f P_cam = Eigen::Affine3f::Identity();
    P_cam.matrix()
        <<
        256, 0., 256, 0.,
        0., 256, 256, 0.,
        0., 0., 1., 0.,
        0, 0, 0, 1;
    Eigen::Matrix3f R_cam;
    R_cam
        <<
        0., -1, 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Isometry3f vehicle2velo;
    vehicle2velo.matrix()
        <<
        1., 0., 0., 1.13,
        0., 1., 0., -0.08,
        0., 0., 1., -1.86,
        0, 0, 0, 1;
    Eigen::Isometry3f vehicle2cam;
    vehicle2cam.matrix()
        <<
        1., 0., 0., -0.39,
        0., 1., 0., 0.0,
        0., 0., 1., -1.2,
        0, 0, 0, 1;
    transform = P_cam * R_cam * vehicle2cam /* * vehicle2velo.inverse() */;
//    std::cout << transform.matrix() << std::endl;
  }

  Eigen::Vector3f operator()(const Eigen::Vector3f &xyz) const {
    Eigen::Vector3f uvw = transform * xyz;
    float u = uvw[0], v = uvw[1], w = uvw[2];
    return {u / w, v / w, w};
  }

  pcl::PointXYZ operator()(const pcl::PointXYZ &xyz) const {
    auto uvw = operator()(Eigen::Vector3f{xyz.x, xyz.y, xyz.z});
    return {uvw[0], uvw[1], uvw[2]};
  }
  Eigen::Affine3f transform;
};

visualization_msgs::MarkerArray::Ptr makeMarkerArray(
    const std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> &aabb,
    const std_msgs::Header &header) {

  visualization_msgs::MarkerArray::Ptr markers(new visualization_msgs::MarkerArray);
  int marker_id = 0;
  for (const auto &bb : aabb) {
    const auto &min_pt = bb.first, &max_pt = bb.second;
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.id = marker_id++;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.1;
    marker.color.g = 1;
    marker.color.a = 1;
    marker.lifetime = ros::Duration(10);
    marker.pose.orientation.w = 1.;
    std::vector<geometry_msgs::Point> points(8);
    points[0].x = min_pt.x;
    points[0].y = min_pt.y;
    points[0].z = min_pt.z;
    points[1].x = max_pt.x;
    points[1].y = min_pt.y;
    points[1].z = min_pt.z;
    points[2].x = max_pt.x;
    points[2].y = max_pt.y;
    points[2].z = min_pt.z;
    points[3].x = min_pt.x;
    points[3].y = max_pt.y;
    points[3].z = min_pt.z;
    points[4].x = min_pt.x;
    points[4].y = min_pt.y;
    points[4].z = max_pt.z;
    points[5].x = max_pt.x;
    points[5].y = min_pt.y;
    points[5].z = max_pt.z;
    points[6].x = max_pt.x;
    points[6].y = max_pt.y;
    points[6].z = max_pt.z;
    points[7].x = min_pt.x;
    points[7].y = max_pt.y;
    points[7].z = max_pt.z;

    marker.points.push_back(points[0]);
    marker.points.push_back(points[1]);
    marker.points.push_back(points[1]);
    marker.points.push_back(points[2]);
    marker.points.push_back(points[2]);
    marker.points.push_back(points[3]);
    marker.points.push_back(points[3]);
    marker.points.push_back(points[0]);
    marker.points.push_back(points[4]);
    marker.points.push_back(points[5]);
    marker.points.push_back(points[5]);
    marker.points.push_back(points[6]);
    marker.points.push_back(points[6]);
    marker.points.push_back(points[7]);
    marker.points.push_back(points[7]);
    marker.points.push_back(points[4]);

    marker.points.push_back(points[0]);
    marker.points.push_back(points[4]);
    marker.points.push_back(points[1]);
    marker.points.push_back(points[5]);
    marker.points.push_back(points[2]);
    marker.points.push_back(points[6]);
    marker.points.push_back(points[3]);
    marker.points.push_back(points[7]);
    markers->markers.push_back(marker);
  }
  return markers;
}

class DetectorNode {
 public:
  explicit DetectorNode(ros::NodeHandle &nh,
                        const std::string &image_topic,
                        const std::string &velo_topic,
                        const std::string &odom_topic,
                        const std::string &target_topic,
                        const std::string &engine_name,
                        bool verbose) : detector_(engine_name, {0}),
                                        verbose_(verbose),
                                        pointcloud_(new pcl::PointCloud<pcl::PointXYZ>) {
    image_sub_ = nh.subscribe<sensor_msgs::Image>(image_topic, 1, &DetectorNode::imageUpdate, this);
    velo_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(velo_topic, 1, &DetectorNode::detectAndPublish, this);
    odom_sub_ = nh.subscribe<std_msgs::Float32MultiArray>(odom_topic, 1, &DetectorNode::odomUpdate, this);
    ROS_INFO_STREAM("Subscribing to " << image_topic << " and " << velo_topic);
    target_pub_ = nh.advertise<std_msgs::Float32MultiArray>(target_topic, 1);

    velo_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cone", 1);
    anno_pub_ = nh.advertise<sensor_msgs::Image>("annotated", 1);
    bbox_pub_ = nh.advertise<visualization_msgs::MarkerArray>("markers", 1);
  }

  void detectAndPublish(const sensor_msgs::PointCloud2::ConstPtr &pcl_msg) {
    if (image_ == nullptr) return;
    pcl::fromROSMsg(*pcl_msg, *pointcloud_);
    if (pointcloud_->empty()) return;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_proj(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : *pointcloud_) {
      pcl_proj->push_back(proj_model_(point));
    }

    std::vector<cv::Rect> res;
    detector_.detect(image_->image, res, verbose_);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> aabbs;
    std_msgs::Float32MultiArray targets;
    auto rotation = Eigen::AngleAxisf(rpy_[2], Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(rpy_[1], Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(rpy_[0], Eigen::Vector3f::UnitX());

    for (const auto &rect : res) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr crude_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      clusters.push_back(crude_cluster);
      if (verbose_) {
        std::cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << std::endl;
      }
      for (auto pt_raw = pointcloud_->begin(), pt_proj = pcl_proj->begin(); pt_raw != pointcloud_->end();
           ++pt_raw, ++pt_proj) {
        if (pt_proj->z > 0 and pt_proj->z < 100 and rect.contains({int(pt_proj->x), int(pt_proj->y)})) {
          crude_cluster->push_back(*pt_raw);
        }
      }
      if (not crude_cluster->empty()) {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(crude_cluster);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.4);
        ec.setMinClusterSize(5);
        ec.setMaxClusterSize(100);
        ec.setSearchMethod(tree);
        ec.setInputCloud(crude_cluster);
        ec.extract(cluster_indices);
        int max_size = 0, argmax = -1;
        for (int i = 0; i < cluster_indices.size(); ++i) {
          int size = cluster_indices[i].indices.size();
          if (size > max_size) {
            max_size = size;
            argmax = i;
          }
        }
        if (argmax != -1) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::copyPointCloud(*crude_cluster, cluster_indices[argmax].indices, *cluster);

          pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
          feature_extractor.setInputCloud(cluster);
          feature_extractor.compute();
          std::pair<pcl::PointXYZ, pcl::PointXYZ> aabb;
          feature_extractor.getAABB(aabb.first, aabb.second);
          if (verbose_) {
            std::cout << "AABB: " << aabb.first << " " << aabb.second << std::endl;
          }
          aabbs.push_back(aabb);
          Eigen::Vector3f center{(aabb.first.x + aabb.second.x) / 2,
                                 (aabb.first.y + aabb.second.y) / 2,
                                 (aabb.first.z + aabb.second.z) / 2};
          center = rotation * center + position_;
          targets.data.push_back(center.x());
          targets.data.push_back(center.y());
          targets.data.push_back(center.z());
        }
      }
    }
    target_pub_.publish(targets);
    if (verbose_) {
      pcl::PointCloud<pcl::PointXYZ> cone;
      for (auto pt_raw = pointcloud_->begin(), pt_proj = pcl_proj->begin(); pt_raw != pointcloud_->end();
           ++pt_raw, ++pt_proj) {
        if (pt_proj->x > 0 and pt_proj->x < 512 and pt_proj->y > 0 and pt_proj->y < 512 and pt_proj->z > 0) {
          cone.push_back(*pt_raw);
        }
      }
      sensor_msgs::PointCloud2 cone_msg;
      pcl::toROSMsg(cone, cone_msg);
      cone_msg.header = pcl_msg->header;
      velo_pub_.publish(cone_msg);

      bbox_pub_.publish(makeMarkerArray(aabbs, pcl_msg->header));

      for (auto &pt_proj : *pcl_proj) {
        if (pt_proj.x > 0 and pt_proj.x < 512 and pt_proj.y > 0 and pt_proj.y < 512 and pt_proj.z > 0) {
          image_->image.at<int>(int(pt_proj.y), int(pt_proj.x), 2) = 255;
        }
      }
      anno_pub_.publish(image_->toImageMsg());
    }
  }

  void imageUpdate(const sensor_msgs::Image::ConstPtr &image_msg) {
    image_ = cv_bridge::toCvCopy(image_msg, "bgr8");
  }

  void odomUpdate(const std_msgs::Float32MultiArray::ConstPtr &odom) {
    for (int i = 0; i < 3; ++i) {
      position_[i] = odom->data[i];
    }
    for (int i = 0; i < 3; ++i) {
      rpy_[i] = odom->data[i + 3] / 180 * M_PI;
    }
  }
 private:
  Detector detector_;
  ProjectionModelCustom proj_model_;
  ros::Subscriber image_sub_, velo_sub_, odom_sub_;
  ros::Publisher target_pub_;
  bool verbose_;
  ros::Publisher anno_pub_, bbox_pub_, velo_pub_;

  cv_bridge::CvImagePtr image_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_;
  Eigen::Vector3f position_ = Eigen::Vector3f::Zero(), rpy_ = Eigen::Vector3f::Zero();
};

int main(int argc, char **argv) {
  cudaSetDevice(DEVICE);
  ros::init(argc, argv, "detector_node");
  ros::NodeHandle nh;
  std::string image_topic, velo_topic, odom_topic, target_topic, engine_path;
  bool verbose;

  nh.getParam("image_topic", image_topic);
  nh.getParam("velo_topic", velo_topic);
  nh.getParam("odom_topic", odom_topic);
  nh.getParam("target_topic", target_topic);
  nh.getParam("engine_path", engine_path);
  nh.getParam("verbose", verbose);

  auto detector = DetectorNode(nh, image_topic, velo_topic, odom_topic, target_topic, engine_path, verbose);

  while (ros::ok()) {
    ros::spinOnce();
  }
  return 0;
}
