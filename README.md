# YOLOv4 + Deep_SORT + person_reID_baseline

## Requirement

__Development Environment: 

* OpenCV
* sklean
* pillow
* numpy 1.15.0
* torch 1.3.0
* tensorflow-gpu 1.13.1
* CUDA 10.0
* Pytorch 0.3+

***

It uses:

* __Detection__: [YOLOv4](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/deep_sort_yolov4) to detect objects on each of the video frames. 
* __Tracking__: [Deep_SORT](https://github.com/layumi/Person_reID_baseline_pytorchhttps://github.com/Qidian213/deep_sort_yolov3) to track those objects over different frames.
* __Re-identification__：[Person_reID_baseline](https://github.com/bubbliiiing/yolov4-kerashttps://github.com/LeonLok/Deep-SORT-YOLOv4) to re-identification person in the saved person images.

## Quick Start

__0.Requirements__

    pip install -r requirements.txt

__1. Download the code to your computer.__
    

    git clone https://github.com/582751489/LY_PDTR.git

__2. Download [[yolov4.weights]](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view) [[Baidu]](https://pan.baidu.com/s/1jRudrrXAS3DRGqT6mL4L3A ) - `mnv6`__ and place it in `PDTR/Main/model_data/`

*Here you can download my trained [[yolo4_weight.h5]](https://pan.baidu.com/s/1JuT4KCUFaE2Gvme0_S37DQ ) - `w17w` weights for detecting person/car/bicycle,etc.*

__3. Convert the Darknet YOLO model to a Keras model:__

```
$ python convert.py model_data/yolov4.cfg model_data/yolov4.weights model_data/yolo.h5
```

__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH] -ids [TRACKING ID]

$ python main.py -c person -i -ids 5./test_video/testvideo.avi
```

__5. Can change [deep_sort_yolov3/yolo.py] `__Line 100__` to your tracking object__

*DeepSORT pre-trained weights using people-ReID datasets only for person*

```
    if predicted_class != args["class"]:
               continue
    
    if predicted_class != 'person' and predicted_class != 'car':
               continue
```

## Train on Market1501 & self-made Date set

*People Re-identification model*

Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

Preparation: Put the images with the same id in one folder. You may use.

```markdown
python prepare.py
```

[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

Download [self-made Dataset](https://drive.google.com/file/d/1VPQ8eGgxsxamdkffW8HJixcH_5eHPRA7/view?usp=sharing) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)

## Trained Model for re-identification

The download link is [Here](https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE).

| Methods                  | Rank@1 | mAP    | Reference                                                    |
| ------------------------ | ------ | ------ | ------------------------------------------------------------ |
| [ResNet-50]              | 88.84% | 71.59% | `python train.py --train_all`                                |
| [DenseNet-121]           | 90.17% | 74.02% | `python train.py --name ft_net_dense --use_dense --train_all` |
| [PCB]                    | 92.64% | 77.47% | `python train.py --name PCB --PCB --train_all --lr 0.02`     |
| [ResNet-50 (fp16)]       | 88.03% | 71.40% | `python train.py --name fp16 --fp16 --train_all`             |
| [ResNet-50 (all tricks)] | 91.83% | 78.32% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5` |

### Test for re-identification

Use trained model to extract feature by

```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```

`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.

### Evaluation for re-identification

```bash
python evaluate.py
```

It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

For more evaluation process, go to this ling [Evaluation for PDTR](https://github.com/582751489/Evaluation_for_PDTR.git)

## Visualization

```python vis.py
python vis.py
```

For visualization, plot the pedestrian table and time table.

## Introduction to main code

__main.py__
It is for pedestrian detection and tracking, there are two main fuction of this code:
1) When order do not indicate a target ID: you can run this code like: python main.py  ./test_video/camera9.mp4 -camera c3 to indicate the path of the video path and the camera ID.
Then the detection and tracking video will be in folder 'output', named 'output'+'camera ID', and pedestrian images will be in folder 'perosn' which consist of 'gallery' of all the pedestrian images with corresponding ID and 'query' of one image for re-identification.
2) When order indicate a target ID: you can run this code like: python main.py  ./test_video/camera9.mp4 -camera c3 -id 12 to indicate the path of the video path the petestrian ID and the camera ID .
Then the detection and tracking video will be in folder 'output', named 'output_reid'+'camera ID', and won't save the pedestrian images.This video will only tracking the target pedestrian with color boxes, ID and path line.

__test.py__
1) You should indicate the re-identificateion model before '--name' and pedestrian path before '--test_dir'.
eg: python test.py  --name PCB --test_dir Z:\pro2\whole\person
According to the folder 'person' to do feature extraction and saving, the saving path will be the root path named 'pytorch_result.mat'.

__demo.py__
1) You should indicate the re-identified target ID before '--query_index' and camera ID before '--camera'.
eg: python demo.py --query_index 12 --camera c1 
It is for re-identification by camparing the feature similarity according to file 'pytorch_result.mat'. It will give two output, one is a photo to show the top 10 most similar person, for later evaluation. And another output is the re-identified result: the Camera ID and pedestrian ID.



## Reference

#### Github:https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/deep_sort_yolov4

#### Github:https://github.com/layumi/Person_reID_baseline_pytorchhttps://github.com/Qidian213/deep_sort_yolov3

#### Github:https://github.com/bubbliiiing/yolov4-kerashttps://github.com/LeonLok/Deep-SORT-YOLOv4

