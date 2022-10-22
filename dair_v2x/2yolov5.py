'''
@File    :   2yolov5.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/10/21 14:58:39
@Desc    :   将DAIR-V2X数据集转换为YOLOV5格式
'''

import argparse
import json
import os
from multiprocessing import Process
from typing import Tuple

from tqdm import tqdm


class Loader:

    def __init__(self, data_dir: str, train_split: float = 0.85) -> None:
        """初始化

        Args:
            data_dir (str): DAIR-V2X 数据集根目录
        """
        self.data_dir = data_dir
        self.__categories = self.__get_categories()
        self.__data_info_path = os.path.join(data_dir, "data_info.json")  # 数据信息
        self.__train_data_info, self.__val_data_info = self.__get_train_val_info(train_split)

    @property
    def train_info(self) -> dict:
        """训练数据信息
        """
        return self.__train_data_info

    @property
    def val_info(self) -> dict:
        """验证数据信息
        """
        return self.__val_data_info

    @property
    def categories(self) -> dict:
        """类别与id对应关系
        """
        return self.__categories

    def __get_categories(self) -> dict:
        """获取类别与id对应关系
        """
        __categories = [
            "car", "truck", "van", "bus", "pedestrian", "cyclist", "tricyclist", "motorcyclist",
            "barrowlist", "trafficcone", "pedestrianignore", "carignore", "otherignore",
            "unknown_movable", "unknown_unmovable"
        ]

        return {_category: _id for _id, _category in enumerate(__categories)}

    def __get_train_val_info(self, train_split) -> Tuple[dict, dict]:
        """分割训练集和验证集
        """
        __data_info = json.loads(open(self.__data_info_path).read())
        train_num = int(len(__data_info) * train_split)
        return __data_info[:train_num], __data_info[train_num:]


class DAIR2COCO(Loader):
    """DAIR-V2X数据集转换为coco格式
    """

    def __init__(self, data_dir: str, train_split: float = 0.85) -> None:
        super(DAIR2COCO, self).__init__(data_dir, train_split)
        self.save_dir = os.path.join(data_dir, "yolov5-labels")

    def __get_annotations(self, annos_dir: str) -> dict:
        """获取标注信息
        """
        return json.loads(open(annos_dir).read())

    def __bbox2xywh(self, bbox: dict, image: dict) -> list:
        """将DAIR-V2X中的bbox[xmin, ymin, xmax, ymax], 转为yolov5的bbox[x_center, y_center, width, height]

        Args:
            bbox (dict): DAIR-V2X的bbox

        Returns:
            list: yolov5的bbox
        """
        return [
            ((float(bbox["xmin"]) + float(bbox["xmax"])) / 2) / float(image["width"]),
            ((float(bbox["ymin"]) + float(bbox["ymax"])) / 2) / float(image["height"]),
            (float(bbox["xmax"]) - float(bbox["xmin"])) / float(image["width"]),
            (float(bbox["ymax"]) - float(bbox["ymin"])) / float(image["height"])
        ]

    def format2coco(self, data_info: dict, save_path: str) -> None:
        """转为COCO格式

        Args:
            ids (list): 图片ids
            json_path (str): annotations json 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        for data in tqdm(data_info):
            file_name = data["image_path"]
            img_id, _ = os.path.splitext(os.path.basename(file_name))
            annos_dir = os.path.join(self.data_dir, data["label_camera_std_path"])
            annos = self.__get_annotations(annos_dir)

            image_msg = {"file_name": file_name, "height": 1080, "width": 1920, "id": img_id}

            with open(os.path.join(save_path, f"{img_id}.txt"), "w") as file:
                for i, item in enumerate(annos, start=1):
                    xywh = self.__bbox2xywh(item["2d_box"], image_msg)
                    category = item["type"].lower()
                    category_id = self.categories[category]
                    line = f"{category_id} {' '.join([str(x) for x in xywh])}"
                    if i == len(annos):
                        file.write(line)
                    else:
                        file.write(line + '\n')

    def processing(self) -> None:
        """处理进程
        """
        os.makedirs(self.save_dir, exist_ok=True)
        # 创建进程
        train_process = Process(
            target=self.format2coco,
            kwargs={
                "data_info": self.train_info,
                "save_path": os.path.join(self.save_dir, "train")
            }
        )
        val_process = Process(
            target=self.format2coco,
            kwargs={
                "data_info": self.val_info,
                "save_path": os.path.join(self.save_dir, "val")
            }
        )
        # 启动进程
        train_process.start()
        val_process.start()


def parse_args():
    parser = argparse.ArgumentParser(description="DAIR-V2X dataset to YOLOv5 format.")
    parser.add_argument('--data_dir', type=str, help='数据位置')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DAIR2COCO(args.data_dir).processing()
