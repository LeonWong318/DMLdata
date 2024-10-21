# Multi-Object Tracking in Large-Scale Object Scenarios Using TrackFormer

This repository provides the large-scale object scenarios implementation of the deep machine learning course project (find the report) which borrows a lot of codes from [TrackFormer](https://github.com/timmeinhardt/trackformer)




## Abstract

This paper explores the use of transfer learning to adapt a multi-object tracking model, pre-trained on the MOT20 dataset for full-body tracking, to a new task of head tracking using the HT21 dataset. The aim is to improve tracking accuracy in high-density scenes where full-body detection is prone to failure due to crowding and occlusion. By fine-tuning the MOT20 model with selected subsets of HT21, we successfully shifted the model's focus from body to head tracking, leading to a significant increase in detection performance, particularly in crowded environments. However, the new model demonstrated limitations in handling head occlusion and consistent identity tracking after occlusions. Additionally, issues such as poor detection of stationary individuals persisted. Overall, the transfer learning approach proved effective, demonstrating its ability to adapt existing models for new tasks with improved accuracy, while also revealing areas for further refinement.


## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluation

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### HT21



```
python3 src/track.py with \
    dataset_name=HT21-14 \
    data_root_dir=data \
    output_dir=data/HT21_14_after_TL \
    write_images=pretty \
    reid \
    obj_detect_checkpoint_file=models/custom_dataset_deformable/checkpoint_epoch_10.pth
```

<center>

| HT21-01     | MOTA         | MOTP           |       IDF1     |     HOTA     |     MTR       |     MLR             |  
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  
| **Train** |     66.77     |     71.109       |     67.692      | 49.46        |      45.51    |      16.456          |  

</center>

## Visualization

You may use the following code to generate the final video based on running the evaluation part.

```
ffmpeg -r 25 -f image2 -i /root/trackformer/data/HT21_14_after_TL/HT21-14/HT21-14-None/%06d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p HT21_14_after_TL.mp4
```


