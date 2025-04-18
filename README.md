### Environment

This code runs the environmeny where **Wan 2.1** can be installed, and then installs the required libraries.

or

```sh
conda env create -f my_env.yaml
```

### Prepare your dataset

You need to manage the training videos as follows:

```
 data_dir
|—— meta.json
|—— input_video
    |—— video_1.mp4
    |—— video_2.mp4
    |—— ...
|—— condition_video
    |—— video_1.mp4
    |—— video_2.mp4
    |—— ...
```
You can refer to meta.json in /val_data/meta.json for an example of the format.

The format of meta.json is as follows:

```json
{"input_video": "./input_video/video_1.mp4", "condition": ".condition_video/video_1.mp4", "caption": "prompt"}
{"input_video": "./input_video/video_2.mp4", "condition": ".condition_video/video_2.mp4", "caption": "prompt"}
...
```

### Train

```sh
accelerate launch --config_file ./default_config.yaml scripts/train_wan_t2v.py \
  --train_batch_size 2 \
  --train_data_dir "./val_data/meta.json" \
  --pretrained_model_name_or_path "Path to pretrained model" \
  --validation_prompt "val prompt" \
  --val_condition_path "path of val condition"
  --gradient_checkpointing --checkpointing_steps 1000 --rank 256 \
  --validation_steps 20 --scale_lr \
  --use_keyframe \
  --output_dir ./CA_addkey 
```

## Thank you




