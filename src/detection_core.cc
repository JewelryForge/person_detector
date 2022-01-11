#include "detection_core.h"

#include <utility>

DetectionCore::DetectionCore(ros::NodeHandle &nh) {
  seg_distance_ = {10, 30, 45, 60};
  cluster_distance_ = {0.52, 1.2, 2, 3.0, 4.5};

  ROS_INFO("[2]+running obstacle_detection_node");
  boxer = new BoundingBoxCalculator;
  boxer->init();

//  nh.param("cluster_topic", cluster_topic, std::string("/kitti/velo/pointcloud"));

  // sub_pc = nh.subscribe("/velodyne_points", 5, &DetectionCore::callback, this);
//  sub_pc = nh.subscribe(cluster_topic, 10, &DetectionCore::callback, this);

//  pub_bbox = nh.advertise<visualization_msgs::MarkerArray>("markers", 10);
//  pub_cluster = nh.advertise<sensor_msgs::PointCloud2>("cluster", 10);
//  pub_lidarMarkerArray = nh.advertise<lidarmarkers_msg::lidarMarkers>("markersArray", 10);

//  ros::spin();
}

DetectionCore::~DetectionCore() {}

void DetectionCore::callback(const sensor_msgs::PointCloud2::ConstPtr &pc_in,
                             std::vector<BoundingBoxCalculator::BoundingBox> &boxes) {
  auto startTime = std::chrono::steady_clock::now();

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*pc_in, *pcl_cloud);


  /** 1.过滤点云  **/
  pcl_cloud = FilterCloud(pcl_cloud);


  // velodyne16离群点去除
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> pcFilter;  //创建滤波器对象
  pcFilter.setInputCloud(pcl_cloud);             //设置待滤波的点云
  pcFilter.setRadiusSearch(0.8);               // 设置搜索半径
  pcFilter.setMinNeighborsInRadius(2);      // 设置一个内点最少的邻居数目
  pcFilter.filter(*pcl_cloud);        //滤波结果存储到cloud_filtered


  /**2.分割平面ransac **/
  // TODO:地面分割效果不是很好，直接滤掉z
  // std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentCloud = SegmentCloud(pcl_cloud);

  /**3.聚类 **/
  /*//地面拟合后不需要去除
  pcl::PassThrough<pcl::PointXYZ> pass_z;
  pass_z.setInputCloud(pcl_cloud);
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(-1.20, 1.0);
  pass_z.filter(*pcl_cloud);
  */

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = Clustering(pcl_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  /**4.计算旋转的bbox**/
  for (auto &cloudCluster: cloudClusters) {
    *cluster_cloud += *cloudCluster;
    BoundingBoxCalculator::BoundingBox tmp_box = boxer->calcBoundingBox(*cloudCluster);
    if (tmp_box.size.x * tmp_box.size.y > 18)
      continue;
    Eigen::Vector2f v1(tmp_box.center.x, tmp_box.center.y);
    // float distance = v1.norm();
    // 约束汽车框的大小
    // if (tmp_box.size.x / tmp_box.size.y > 4 && tmp_box.size.y < 0.4)
    //     continue;
    // if (tmp_box.size.x < 5.1 && tmp_box.size.y < 3.1
    //     && tmp_box.size.x > 1.8 && tmp_box.size.y > 0.8 )
    boxes.push_back(tmp_box);
  }
//  int box_count = 0;
//  visualization_msgs::MarkerArray markers;
//  lidarmarkers_msg::lidarMarkers lidarmarkers;
//  lidarmarkers.header = pc_in->header;
  // markers.header= pc_in->header;
//  std::string frameId = pc_in->header.frame_id;
//  markers.markers.clear();
//  for (auto box: boxes) {
//    visualization_msgs::Marker marker1;
//    marker1.header = pc_in->header;
//    boxer->draw_box(box, box_count++, marker1, 1.1, frameId);
//    markers.markers.push_back(marker1);
//  }
//  lidarmarkers.markerArray = markers;
//  pub_bbox.publish(markers);
//  pub_lidarMarkerArray.publish(lidarmarkers);

//  sensor_msgs::PointCloud2 cluster_mag;
//  pcl::toROSMsg(*cluster_cloud, cluster_mag);
  // cluster_mag.header.stamp = pc_in->header.stamp;
//  cluster_mag.header = pc_in->header;
//  cluster_mag.header.frame_id = pc_in->header.frame_id;
//  pub_cluster.publish(cluster_mag);

//  auto endTime = std::chrono::steady_clock::now();
//  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  // std::cout << "Total took " << elapsedTime.count() << " milliseconds" << std::endl;

}

visualization_msgs::MarkerArray
DetectionCore::makeMarkerArray(const std::vector<BoundingBoxCalculator::BoundingBox> &bboxes,
                               const std_msgs::Header &header) {

  visualization_msgs::MarkerArray markers;
  markers.markers.clear();
  int box_count = 0;
  for (const auto &box: bboxes) {
    visualization_msgs::Marker marker1;
    marker1.header = header;
    boxer->draw_box(box, box_count++, marker1, 1.1, header.frame_id);
    markers.markers.push_back(marker1);
  }
  return markers;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
DetectionCore::FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  float filterRes = 0.2;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(filterRes, filterRes, filterRes);
  vg.filter(*cloudFiltered);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudRegion(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::CropBox<pcl::PointXYZ> region(true);
  // Eigen::Vector4f minPoint = (-10, -5, -2, 1);
  // Eigen::Vector4f maxPoint = (30, 8, 1, 1);
  region.setMin(Eigen::Vector4f(-100, -50, -1.4, 1));
  region.setMax(Eigen::Vector4f(100, 50, 0.5, 1));
  region.setInputCloud(cloudFiltered);
  region.filter(*cloudRegion);

  // 移除车顶点云
  /*
  std::vector<int> indices;
  pcl::CropBox<pcl::PointXYZ> roof(true);
  roof.setMin(Eigen::Vector4f(-1.58, -1.7, -2, 1));
  roof.setMax(Eigen::Vector4f(2.6, 1.7, 4, 1));
  roof.setInputCloud(cloudRegion);
  roof.filter(indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for(int point : indices)
      inliers->indices.push_back(point);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloudRegion);
  extract.setIndices(inliers);
  // extract.setIndices(indices);
  extract.setNegative(true);
  extract.filter(*cloudRegion);
  */

  return cloudRegion;
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>
DetectionCore::SegmentCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
  pcl::ModelCoefficients::Ptr coefficients{new pcl::ModelCoefficients};

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(25);
  seg.setDistanceThreshold(0.3);

  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() == 0) {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr obstCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr planecloud(new pcl::PointCloud<pcl::PointXYZ>);

  for (int index: inliers->indices)
    planecloud->points.push_back(cloud->points[index]);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*obstCloud);

  std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segResult(obstCloud, planecloud);

  return segResult;
}
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
DetectionCore::Clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_pc_array(5);

  for (size_t i = 0; i < segment_pc_array.size(); i++) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    segment_pc_array[i] = tmp;
  }

  for (size_t i = 0; i < cloud->points.size(); ++i) {
    pcl::PointXYZ current_point;
    current_point.x = cloud->points[i].x;
    current_point.y = cloud->points[i].y;
    current_point.z = cloud->points[i].z;

    float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));
    if (origin_distance >= 120) {
      continue;
    }
    if (origin_distance < seg_distance_[0]) {
      segment_pc_array[0]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[1]) {
      segment_pc_array[1]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[2]) {
      segment_pc_array[2]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[3]) {
      segment_pc_array[3]->points.push_back(current_point);
    } else {
      segment_pc_array[4]->points.push_back(current_point);
    }

  }

  for (size_t i = 0; i < segment_pc_array.size(); i++) {
    cluster_segment(segment_pc_array[i], cluster_distance_[i], clusters);
  }

  return clusters;
}
void DetectionCore::cluster_segment(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                    double in_max_cluster_distance,
                                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(in_max_cluster_distance);
  ec.setMinClusterSize(8);
  ec.setMaxClusterSize(500);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(clusterIndices);

  pcl::PointXYZ minPt;
  pcl::PointXYZ maxPt;

  for (pcl::PointIndices getIndices: clusterIndices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (int index: getIndices.indices)
      cloudCluster->points.push_back(cloud->points[index]);

    pcl::getMinMax3D(*cloudCluster, minPt, maxPt);
    float length = maxPt.x - minPt.x;
    float width = maxPt.y - minPt.y;
//    float highth = maxPt.z - minPt.z;

    if (length < 5 && width < 5 && length >= 0.5 && width >= 0.5) {
      // reSampleCloud(cloudCluster, 0.1);
      cloudCluster->width = cloudCluster->points.size();
      cloudCluster->height = 1;
      cloudCluster->is_dense = true;
      clusters.push_back(cloudCluster);
    }
  }
}

void DetectionCore::reSampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pts_in, float offset) {
  pcl::PointXYZ point;
  int num = pts_in->points.size();
  for (int i = 0; i < num; ++i) {
    point.x = pts_in->points[i].x + offset;
    point.y = pts_in->points[i].y + offset;
    point.z = pts_in->points[i].z;
    pts_in->points.push_back(point);
  }
  for (int i = 0; i < num; ++i) {
    point.x = pts_in->points[i].x - offset;
    point.y = pts_in->points[i].y - offset;
    point.z = pts_in->points[i].z;
    pts_in->points.push_back(point);
  }

}

LidarDetector::LidarDetector() {
  seg_distance_ = {10, 30, 45, 60};
  cluster_distance_ = {0.52, 1.2, 2, 3.0, 4.5};

  boxer = new BoundingBoxCalculator;
  boxer->init();
}

std::vector<BoundingBoxCalculator::BoundingBox> LidarDetector::detect(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc) {
  std::cout << "inner0" << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = FilterCloud(pc);

  // velodyne16离群点去除
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> pcFilter;  //创建滤波器对象
  pcFilter.setInputCloud(pcl_cloud);             //设置待滤波的点云
  pcFilter.setRadiusSearch(0.8);               // 设置搜索半径
  pcFilter.setMinNeighborsInRadius(2);      // 设置一个内点最少的邻居数目
  pcFilter.filter(*pcl_cloud);        //滤波结果存储到cloud_filtered
  std::cout << "inner1" << std::endl;


  /**2.分割平面ransac **/
  // TODO:地面分割效果不是很好，直接滤掉z
  // std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentCloud = SegmentCloud(pcl_cloud);

  /**3.聚类 **/
  /*//地面拟合后不需要去除
  pcl::PassThrough<pcl::PointXYZ> pass_z;
  pass_z.setInputCloud(pcl_cloud);
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(-1.20, 1.0);
  pass_z.filter(*pcl_cloud);
  */

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = Clustering(pcl_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::cout << "inner2" << std::endl;

  /**4.计算旋转的bbox**/
  std::vector<BoundingBoxCalculator::BoundingBox> boxes;
  for (auto &cloudCluster: cloudClusters) {
    *cluster_cloud += *cloudCluster;
    BoundingBoxCalculator::BoundingBox tmp_box = boxer->calcBoundingBox(*cloudCluster);
    if (tmp_box.size.x * tmp_box.size.y > 18)
      continue;
    Eigen::Vector2f v1(tmp_box.center.x, tmp_box.center.y);
    // float distance = v1.norm();
    // 约束汽车框的大小
    // if (tmp_box.size.x / tmp_box.size.y > 4 && tmp_box.size.y < 0.4)
    //     continue;
    // if (tmp_box.size.x < 5.1 && tmp_box.size.y < 3.1
    //     && tmp_box.size.x > 1.8 && tmp_box.size.y > 0.8 )
    boxes.push_back(tmp_box);
  }
  std::cout << "inner3" << std::endl;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarDetector::FilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  float filterRes = 0.2;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(filterRes, filterRes, filterRes);
  vg.filter(*cloudFiltered);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudRegion(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::CropBox<pcl::PointXYZ> region(true);
  // Eigen::Vector4f minPoint = (-10, -5, -2, 1);
  // Eigen::Vector4f maxPoint = (30, 8, 1, 1);
  region.setMin(Eigen::Vector4f(-100, -50, -1.2, 1));
  region.setMax(Eigen::Vector4f(100, 50, 0.5, 1));
  region.setInputCloud(cloudFiltered);
  region.filter(*cloudRegion);

  // 移除车顶点云
  /*
  std::vector<int> indices;
  pcl::CropBox<pcl::PointXYZ> roof(true);
  roof.setMin(Eigen::Vector4f(-1.58, -1.7, -2, 1));
  roof.setMax(Eigen::Vector4f(2.6, 1.7, 4, 1));
  roof.setInputCloud(cloudRegion);
  roof.filter(indices);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  for(int point : indices)
      inliers->indices.push_back(point);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloudRegion);
  extract.setIndices(inliers);
  // extract.setIndices(indices);
  extract.setNegative(true);
  extract.filter(*cloudRegion);
  */

  return cloudRegion;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
LidarDetector::Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_pc_array(5);

  for (auto &seg: segment_pc_array) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    seg = tmp;
  }

  for (size_t i = 0; i < cloud->points.size(); ++i) {
    pcl::PointXYZ current_point;
    current_point.x = cloud->points[i].x;
    current_point.y = cloud->points[i].y;
    current_point.z = cloud->points[i].z;

    float origin_distance = std::hypot(current_point.x, current_point.y);
    if (origin_distance >= 120) {
      continue;
    }
    if (origin_distance < seg_distance_[0]) {
      segment_pc_array[0]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[1]) {
      segment_pc_array[1]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[2]) {
      segment_pc_array[2]->points.push_back(current_point);
    } else if (origin_distance < seg_distance_[3]) {
      segment_pc_array[3]->points.push_back(current_point);
    } else {
      segment_pc_array[4]->points.push_back(current_point);
    }

  }

  for (size_t i = 0; i < segment_pc_array.size(); i++) {
    cluster_segment(segment_pc_array[i], cluster_distance_[i], clusters);
  }

  return clusters;
}

void LidarDetector::cluster_segment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                    double in_max_cluster_distance,
                                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(in_max_cluster_distance);
  ec.setMinClusterSize(8);
  ec.setMaxClusterSize(500);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(clusterIndices);

  pcl::PointXYZ minPt;
  pcl::PointXYZ maxPt;

  for (const pcl::PointIndices &getIndices: clusterIndices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (int index: getIndices.indices)
      cloudCluster->points.push_back(cloud->points[index]);

    pcl::getMinMax3D(*cloudCluster, minPt, maxPt);
    float length = maxPt.x - minPt.x;
    float width = maxPt.y - minPt.y;
    float highth = maxPt.z - minPt.z;

    if (length < 5 && width < 5 && length >= 0.5 && width >= 0.5) {
      // reSampleCloud(cloudCluster, 0.1);
      cloudCluster->width = cloudCluster->points.size();
      cloudCluster->height = 1;
      cloudCluster->is_dense = true;
      clusters.push_back(cloudCluster);
    }
  }
}

void LidarDetector::draw_box(const BoundingBoxCalculator::BoundingBox &box,
                             const int &marker_id,
                             visualization_msgs::Marker &marker,
                             float scale,
                             std::string cloudId) {
  boxer->draw_box(box, marker_id, marker, scale, std::move(cloudId));
}