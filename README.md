# programmers_dev_day3 - Image classification

## DarkNet53 binary classification

### Dataset : Dog vs Cat

[link](https://www.kaggle.com/competitions/dogs-vs-cats/data)

### Model : Darknet53

- pretrained weights : [link](https://drive.google.com/file/d/1keZwVIfcWmxfTiswzOKUwkUz2xjvTvfm/view)

### Use

#### Prerequisite

Recommend to use virtual environments(Docker or conda)

```bash
pip install -r requirements.txt
```

#### Run

```bash
python main.py --mode ${train/eval} --output_dir ${output_path} --checkpoint ${pretrained_weight_path} --data ${data_directory_path}
```

#### Use Tensorboard

```bash
tensorboard --logdir=${output_path} --port 8888
```

you can visualize the loss graph in localhost:8888




