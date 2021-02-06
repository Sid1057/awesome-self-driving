# Awesome self-driving [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

 A list of awesome stuff for awesome self-driving stuff.

## Frameworks & libraries

### Computer vision

- [OpenCV](https://github.com/opencv/opencv)
- [Visionworks](https://developer.nvidia.com/embedded/visionworks)

### Machine learning

- [PyTorch](http://pytorch.org/)
- [Torchvision](https://github.com/pytorch/vision)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [CNTK](https://github.com/microsoft/CNTK)
- [ONNX](https://github.com/onnx/onnx)

#### Inference

- [Torch2TRT](https://github.com/NVIDIA-AI-IOT/torch2trt)

### 3D data processing

- [Open3D](http://open3d.org/)

### Simulators

- [Carla](https://github.com/carla-simulator/carla)
- [Carla map editor](https://github.com/carla-simulator/carla-map-editor)
- [LGSVL](https://github.com/lgsvl/simulator)
- [AirSim](https://github.com/microsoft/AirSim)


## Topics

### OS 

#### UNIX stuff

- [Python UNIX service implementation](https://github.com/torfsen/service)

#### OS ENviroment

- [Pylot](https://github.com/erdos-project/pylot)
- [ROS2](https://github.com/ros2/ros2)
- [ROS](https://github.com/ros/ros)
- [Apollo](https://github.com/ApolloAuto/apollo)
- [Autoware](https://github.com/Autoware-AI/autoware.ai)

#### Comminications

- [ZCM](https://github.com/ZeroCM/zcm)

### Sensors / Hardware

- [List of GStreamer vision plugins](https://github.com/joshdoe/gst-plugins-vision)
- [Pylon GStreamer plugin](https://github.com/MattsProjects/pylon_gstreamer)
- [NVIDIA Jetson IMX477 RPIv3 driver](https://github.com/RidgeRun/NVIDIA-Jetson-IMX477-RPIV3)
- [ZCM GStreamer plugin](https://github.com/ZeroCM/zcm-gstreamer-plugins)
- [Intel RealSense library](https://github.com/IntelRealSense/librealsense)

- [Charuco camera calibration](https://github.com/nullboundary/CharucoCalibration)
- [Lidar to camera calibration](https://github.com/ankitdhall/lidar_camera_calibration)

### Perception

#### Lidar

##### Object detection

- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)


#### Camera

##### Object detection

- [MMDetection toolbox](https://github.com/open-mmlab/mmdetection)
- [MMDetection 3D toolbox](https://github.com/open-mmlab/mmdetection3d)
- [Deep learning object detection SOTA](https://github.com/hoya012/deep_learning_object_detection)
- [PyTorch tutorial for object detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)

##### Human pose estimation

- [MMPose toolbox](https://github.com/open-mmlab/mmpose)

##### Action recognition

- [MMAction toolbox](https://github.com/open-mmlab/mmaction2)

##### Segmentation

- [MMSegmentation toolbox](https://github.com/open-mmlab/mmsegmentation)

##### Depth estimation

- [Simple depth estimation using CNN](https://github.com/ArpitaSTugave/Depth-Estimation-using-CNN)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
- [MiDaS](https://github.com/intel-isl/MiDaS)
- [Facebook research consistent depth](https://github.com/facebookresearch/consistent_depth)

##### Stereo Vision

- [Dense depth](https://github.com/ialhashim/DenseDepth)
- [PSMNet](https://github.com/JiaRenChang/PSMNet)
- [Semi-global blockmatching CUDA implementation](https://github.com/fixstars/libSGM)
- [Yet another deep stereovision](https://github.com/vijaykumar01/stereo_matching)

##### Depth obstacles segmentation

- [V-Disp](https://github.com/thinkbell/v_disparity)

##### Structure from motion

- [COLMAP](https://github.com/colmap/colmap)

##### Visual tracking

- [GPL TLD C++ implementation](https://github.com/arthurv/OpenTLD)

##### NN Tuning and explanation

- [GradCam](https://github.com/jacobgil/pytorch-grad-cam)

### Control
...

### Localization

- [PythonRobotics](https://atsushisakai.github.io/PythonRobotics/)
- [Laika: python GNSS processing library](https://github.com/commaai/laika)

#### SLAM and odometry

- [Awesome SLAM datasets](https://github.com/youngguncho/awesome-slam-datasets)
- [OpenCV RGBD Odometry](https://github.com/tzutalin/OpenCV-RgbdOdometry)

### Planning

#### RL

- [Min Carla environment](https://github.com/mcemilg/min-carla-env)
- [Highway env](https://github.com/eleurent/highway-env)
- [RL for Decision-Making for self-driving-cars](https://github.com/chauvinSimon/Reinforcement-Learning-for-Decision-Making-in-self-driving-cars)
- [Learning by cheating](https://github.com/dianchen96/LearningByCheating)
- [Learning by cheating](https://github.com/bradyz/2020_CARLA_challenge)
- [Jetson reinforcement learning](https://github.com/dusty-nv/jetson-reinforcement)

## Visualization

- [Carla Viz plugin](https://github.com/carla-simulator/carlaviz)
- [StreetScape.gl](https://github.com/uber/streetscape.gl)
- [XVIZ](https://github.com/uber/xviz)


## Datasets

### Real datasets

- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [KITTI 360](http://www.cvlibs.net/datasets/kitti-360)
- [CityScapes](https://www.cityscapes-dataset.com/)
- [A2D2](https://www.a2d2.audi/a2d2/en.html)
- [German traffic signs](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=about)
- [Icevision traffic lights](https://github.com/icevision/annotations)
- [Awesome public datasets](https://github.com/awesomedata/awesome-public-datasets)

### Synthetics datasets

- [vkitti v2](https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/)
- [Carla dataset runner](https://github.com/AlanNaoto/carla-dataset-runner)

### Tools

- [Awesome list of data labeling tools](https://github.com/heartexlabs/awesome-data-labeling)
- [CVAT](https://github.com/openvinotoolkit/cvat)
- [Semi-auto image annotataion](https://github.com/virajmavani/semi-auto-image-annotation-tool)
- [Open images downloader](https://github.com/harshilpatel312/open-images-downloader)
- [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts)
- [3D labelling tools](https://github.com/CPFL/3d_labeling_tools)
- [Label IMG](https://github.com/tzutalin/labelImg)

### GANs, augmentation, etc.

- [Image2Image papers](https://github.com/lzhbrian/image-to-image-papers)
- [Vid2Vid](https://github.com/NVIDIA/vid2vid)
- [Summer2Winter stylee transfering](https://github.com/hoya012/fast-style-transfer-tutorial-pytorch)
- [Fast photo style](https://github.com/NVIDIA/FastPhotoStyle)


## Challenges

- [Carla AD Challenge](https://leaderboard.carla.org/)
- [Udacity CarND Advanced Lane-Lines Challenge](https://github.com/udacity/CarND-Advanced-Lane-Lines)


## Awesome information

### Papers research

- [Papers With Code (always SOTA)](https://paperswithcode.com/)
- [Awesome bibliography research in autonomous driving](https://github.com/chauvinSimon/My_Bibliography_for_Research_on_Autonomous_Driving)
- [SOTA in self-driving cars from CVLIBS](https://arxiv.org/abs/1704.05519)

## Blogs, Youtube, magazines

- [List of data science blogs](https://github.com/MLWhiz/data_science_blogs)
- [PyImageSearch](https://www.pyimagesearch.com/blog/)
- [Two Minute Papers](https://www.youtube.com/user/keeroyz)
- [Computer Vision Foundation](https://www.youtube.com/channel/UC0n76gicaarsN_Y9YShWwhw)
- [JetsonHacks](https://www.jetsonhacks.com/)
- [Lex Fridman](https://www.youtube.com/user/lexfridman)
- [GreenTheOnly](https://www.youtube.com/user/greentheonly)
- [Self-driving woman](https://www.youtube.com/channel/UCgsNCNMLicldLnzIQuIlyhA)
- [Sid1057](https://www.youtube.com/channel/UCtTnP2N39ZJtdKt9i5u7meg)
- [DeepScale AI](https://www.youtube.com/channel/UCJA94iWh_d1VwLIXX260erQ)
- [Comma AI](https://www.youtube.com/channel/UCwgKmJM4ZJQRJ-U5NjvR2dg)
- [Learn OpenCV](https://www.learnopencv.com/)


### Courses

- [Self-driving cars specialization by University of Toronto](https://www.coursera.org/specializations/self-driving-cars?)
- [List of awesome courses](https://github.com/prakhar1989/awesome-courses)

### Pretrained models

- [Pretrained models from OpenVINO](https://github.com/openvinotoolkit/open_model_zoo)

### Basic

- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
- [DeepLearning papers roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)

### Other

- [Awesome list of 3D reconstruction methods](https://github.com/openMVG/awesome_3DReconstruction_list)
- [Awesome CARLA list](https://github.com/Amin-Tgz/awesome-CARLA)


# TO SORT

- [vision-for-action](https://github.com/intel-isl/vision-for-action)
- [Converter PyTorch networks to TensorRT](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [Jetson inference](https://github.com/dusty-nv/jetson-inference)



# TODO

- [ ] Order
- [ ] README STructuire
- [ ] Programming Languages
- [ ] Control section
- [ ] Card design
- [ ] Web Version
- [ ] Awesome contribution
- [ ] More Courses
- [ ] More Books
