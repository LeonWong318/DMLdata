# Train TrackFormer

The settings for each dataset are specified in the respective configuration files, e.g., `cfgs/train_ht21.yaml`. The following train commands produced the pre-trained model files mentioned in [docs/INSTALL.md](INSTALL.md).

## HT21 and run with multiple GPUs

In this case, we use 6 Nvidia 2080Ti GPUs in almost 3 hours.

```
python3 -m torch.distributed.launch --nproc_per_node=6 --use_env src/train.py with \
    ht21 \
    deformable \
    multi_frame \
    tracking \
    resume=models/custom_dataset_deformable/checkpoint_epoch_10.pth \
    output_dir=models/custom_dataset_deformable \
    mot_path_train=data/HT21 \
    mot_path_val=data/HT21 \
    train_split=HT21_train_1_coco \
    val_split=HT21_val_1_coco \
    epochs=10
```


