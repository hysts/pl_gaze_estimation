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

### MPIIFaceGaze

```bash
test_id=0
python train.py --configs configs/examples/mpiifacegaze.yaml \
                --options SCHEDULER.WARMUP.EPOCHS 3 \
                          EXPERIMENT.TEST_ID ${test_id} \
                          EXPERIMENT.OUTPUT_DIR exp0000/$(printf %02d ${test_id})
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
