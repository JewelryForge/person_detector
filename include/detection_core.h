#ifndef DETECTION_CORE_H
#define DETECTION_CORE_H
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>
#include <ros/ros.h>
#include <ros/package.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Header.h>

#include <sensor_msgs/PointCloud2.h>
#include "box.h"
// #include "lidarmarkers_msg/lidarMarkers.h"

class DetectionCore {
 private:
//  ros::Subscriber sub_pc;
//  ros::Publisher pub_bbox;
//  ros::Publisher pub_cluster;
//  ros::Publisher pub_lidarMarkerArray;

//  std::string cluster_topic;

  BoundingBoxCalculator *boxer;

  pcl::PointCloud<pcl::PointXYZ>::Ptr FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,
            pcl::PointCloud<pcl::PointXYZ>::Ptr> SegmentCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  void cluster_segment(pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc,
                       double in_max_cluster_distance,
                       std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters);

  void reSampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pts_in, float offset);
  std::vector<double> seg_distance_, cluster_distance_;

 public:
  DetectionCore(ros::NodeHandle &nh);
  ~DetectionCore();
  void callback(const sensor_msgs::PointCloud2ConstPtr &pc_in,
                std::vector<BoundingBoxCalculator::BoundingBox> &boxes);
  visualization_msgs::MarkerArray makeMarkerArray(const std::vector<BoundingBoxCalculator::BoundingBox> &,
                                                  const std_msgs::Header &);
};

class LidarDetector {
 private:
  BoundingBoxCalculator *boxer;

  pcl::PointCloud<pcl::PointXYZ>::Ptr FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void cluster_segment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       double in_max_cluster_distance,
                       std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters);

  std::vector<double> seg_distance_, cluster_distance_;

 public:
  LidarDetector();
  ~LidarDetector() = default;
  std::vector<BoundingBoxCalculator::BoundingBox> detect(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc);
  void draw_box(const BoundingBoxCalculator::BoundingBox &box,
                const int &marker_id,
                visualization_msgs::Marker &marker,
                float scale,
                std::string cloudId);
};

#endif // DETECTION_CORE_H
