# 交通数据集格式转换脚本

## DAIR-V2X

&emsp;&emsp;将 DAIR-V2X 数据集转为[COCO](dair_v2x/2coco.py)、[YOLOv5](dair_v2x/2yolov5.py)、[PP-Vehicle](dair_v2x/2pp_vehicle.py)格式

### 用法

&emsp;&emsp;将`image`、`label`、`data_info.json`放在同级目录下，输入以下命令：

```python
# 2yolov5.py 2pp_vehicle.py 同理
python 2coco.py --data_dir=your_dataset_dir
```

## TT-100K

&emsp;&emsp;将 TT-100K 数据集转为[COCO](tt100k/2coco.py)格式

### 用法

&emsp;&emsp;下载 2021 版 TT-100K 并将 2016 版 `annotations.json` 放至同级（2021版的标注是2016版的补充，但不能单独使用），输入以下命令：

```python
python 2coco.py --data_dir=your_dataset_dir
```

