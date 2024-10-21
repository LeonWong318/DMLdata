# Multi-Object Tracking in Large-Scale Object Scenarios Using TrackFormer

This repository provides the large-scale object scenarios implementation of the deep machine learning course project (find the report) which borrows a lot of codes from [TrackFormer](https://github.com/timmeinhardt/trackformer)




## Abstract

This paper explores the use of transfer learning to adapt a multi-object tracking model, pre-trained on the MOT20 dataset for full-body tracking, to a new task of head tracking using the HT21 dataset. The aim is to improve tracking accuracy in high-density scenes where full-body detection is prone to failure due to crowding and occlusion. By fine-tuning the MOT20 model with selected subsets of HT21, we successfully shifted the model's focus from body to head tracking, leading to a significant increase in detection performance, particularly in crowded environments. However, the new model demonstrated limitations in handling head occlusion and consistent identity tracking after occlusions. Additionally, issues such as poor detection of stationary individuals persisted. Overall, the transfer learning approach proved effective, demonstrating its ability to adapt existing models for new tasks with improved accuracy, while also revealing areas for further refinement.


## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate TrackFormer

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### HT21

#### Private detections

```
python src/track.py with reid
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     74.2     |     71.7       |     849      | 177        |      7431    |      78057          |  1449        |
| **Test**  |     74.1     |     68.0       |    1113      | 246        |     34602    |     108777          |  2829        |

</center>

#### Public detections (DPM, FRCNN, SDP)

```
python src/track.py with \
    reid \
    tracker_cfg.public_detections=min_iou_0_5 \
    obj_detect_checkpoint_file=models/mot17_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     64.6     |     63.7       |    621       | 675        |     4827     |     111958          |  2556        |
| **Test**  |     62.3     |     57.6       |    688       | 638        |     16591    |     192123          |  4018        |

</center>


