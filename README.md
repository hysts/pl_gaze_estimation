# pl_gaze_estimation

Training code for gaze estimation models using MPIIGaze, MPIIFaceGaze, and ETH-XGaze.

## Installation

```bash
pip install -r requirements.txt
```

For docker environment, see [here](#docker-environment).

## Usage

The basic usage is as follows:
```bash
python train.py --configs /path/to/your/config.yaml
```

Multiple config files can be specified to set new variables or overwrite previously
defined values.
You can also overwrite previously defined values by listing the key/value
pairs after the `--options` flags.

```
usage: train.py [-h] --configs [CONFIGS ...] [--options ...]

optional arguments:
  -h, --help            show this help message and exit
  --configs [CONFIGS ...]
                        Paths to config files.
  --options ...         Variables to overwrite. (optional)
```

### MPIIGaze

```bash
test_id=0
python train.py --configs configs/examples/mpiigaze_lenet.yaml \
                --options EXPERIMENT.TEST_ID ${test_id} \
                          EXPERIMENT.OUTPUT_DIR exp0000/$(printf %02d ${test_id})
```

You need to download and preprocess the dataset before training by running the following commands:
```bash
bash scripts/download_mpiigaze_dataset.sh
python scripts/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/
```

### MPIIFaceGaze

```bash
test_id=0
python train.py --configs configs/examples/mpiifacegaze.yaml \
                --options SCHEDULER.WARMUP.EPOCHS 3 \
                          EXPERIMENT.TEST_ID ${test_id} \
                          EXPERIMENT.OUTPUT_DIR exp0000/$(printf %02d ${test_id})
```

You need to download and preprocess the dataset before training by running the following commands:
```bash
bash scripts/download_mpiifacegaze_dataset.sh
python scripts/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/
```

### ETH-XGaze

```bash
python train.py \
    --config configs/examples/eth_xgaze.yaml \
    --options \
        VAL.VAL_INDICES "[1, 23, 24, 35, 38, 46, 58, 63, 70, 78]" \
        SCHEDULER.EPOCHS 15 \
        SCHEDULER.MULTISTEP.MILESTONES "[10, 13, 14]" \
        DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP true \
        EXPERIMENT.OUTPUT_DIR exp0000
```

## Docker Environment
### Build
```bash
docker-compose build train
```

### Train
```bash
docker-compose run --rm -u $(id -u):$(id -g) -v /path/to/datasets:/datasets train python train.py --configs /path/to/your/config.yaml
```

## Results on ETH-XGaze dataset

All the results in the table below are of a single run.
In these experiments, the data of subjects with id 1, 23, 24, 35, 38, 46, 58, 63, 70, and 78
were used as the validation data.
The models were trained for 15 epochs with a multi-step learning rate schedule.
The learning rate was multiplied by 0.1 at epochs 10, 13, and 14.

|  Model             | hflip | GPU    | precision | batch size | optimizer | lr     | weight decay | training time | val angle error | val loss |
|--------------------|-------|--------|-----------|------------|-----------|--------|--------------|---------------|-----------------|----------|
| EfficientNet-lite0 | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |      4h42m    | 5.330           | 0.06970  |
| EfficientNet-b0    | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |      5h58m    | 5.139           | 0.06672  |
| ResNet18           | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |      4h04m    | 4.878           | 0.06427  |
| ResNet50           | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |      8h42m    | 4.720           | 0.06087  |
| HRNet-18           | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |     21h56m    | 4.657           | 0.05937  |
| ResNeSt26d         | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |      8h02m    | 4.409           | 0.05678  |
| RegNetY160         | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |   1d05h30m    | 4.377           | 0.05638  |
| Swin-S             | yes   | V100   | 32        | 64         | Adam      | 0.0001 | 0            |   1d00h08m    | 4.318           | 0.05629  |
| HRNet-64           | yes   | V100x8 | 16        | 64         | AdamW     | 0.0008 | 0.05         |      3h11m    | 4.302           | 0.05523  |
| ResNeSt269e        | yes   | V100x8 | 16        | 56         | AdamW     | 0.0008 | 0.05         |      5h31m    | 4.045           | 0.05200  |

## Related repos

- https://github.com/hysts/pytorch_mpiigaze
- https://github.com/hysts/pytorch_mpiigaze_demo

## References

- Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." In European Conference on Computer Vision (ECCV), 2020. [arXiv:2007.15837](https://arxiv.org/abs/2007.15837), [Project Page](https://ait.ethz.ch/projects/2020/ETH-XGaze/), [GitHub](https://github.com/xucong-zhang/ETH-XGaze), [Leaderboard](https://competitions.codalab.org/competitions/28930)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
- Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)
