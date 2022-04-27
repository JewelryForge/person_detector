# person_detector

基于yolov5和聚类的行人检测

## 环境

Ubuntu18.04 / opencv3.4 / pcl1.7 / cuda10.0 / cudnn7.6.5 / tensorrt7.0.0 / nvinfer7.0.0

## demo

```bash
roslaunch person_detector detector_node.launch
```

launch中的参数：

- velo_topic：订阅的激光话题
- image_topic：订阅的图像话题
- odom_topic: 订阅的里程计信息$(x, y, z, r, p, y)$
- target_topic: 输出的话题名，目标在世界坐标系下的位置 
- engine_path：yolov5的模型文件
- verbose：是否输出结果及发布可视化结果相关话题

（可能）发出的话题：

- annotated：有标注的图像
- markers：检测到的人物的AABB

