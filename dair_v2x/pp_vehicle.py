"""
@File    :   pp_vehicle.py
@Version :   2.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/10/21 11:14:24
@Desc    :   将DAIR-V2X数据集转换为coco格式,用于PP-Vehicle
"""

import argparse
import json
import os
from multiprocessing import Process
from typing import Tuple

from tqdm import tqdm


class Loader:
    """加载数据集标签信息

    将 car, truck, van, bus 类别都设置为vehicle;
    忽略 tricyclist, barrowlist, pedestrianignore,
    carignore, otherignore, unknown_movable, unknown_unmovable 类别
    """

    def __init__(self, data_dir: str, train_split: float = 0.85) -> None:
        """初始化

        Args:
            data_dir (str): DAIR-V2X 数据集根目录
        """
        self.data_dir = data_dir
        self.__categories, self.__vhicles = self.__get_categories()
        self.__data_info_path = os.path.join(data_dir,
                                             "data_info.json")  # 数据信息
        self.__train_data_info, self.__val_data_info = self.__get_train_val_info(
            train_split)

    @property
    def train_info(self) -> dict:
        """训练数据信息"""
        return self.__train_data_info

    @property
    def val_info(self) -> dict:
        """验证数据信息"""
        return self.__val_data_info

    @property
    def categories(self) -> dict:
        """类别与id对应关系"""
        return self.__categories

    @property
    def vhicles(self) -> list:
        """机动车类别"""
        return self.__vhicles

    @staticmethod
    def __get_categories() -> dict:
        """获取类别与id对应关系"""
        __categories = [
            "vehicle",
            "pedestrian",
            "cyclist",
            "motorcyclist",
            "barrowlist",
        ]

        return {_category: _id
                for _id, _category in enumerate(__categories)}, [
                    "car",
                    "truck",
                    "van",
                    "bus",
        ]

    def __get_train_val_info(self, train_split) -> Tuple[dict, dict]:
        """分割训练集和验证集"""
        __data_info = json.loads(open(self.__data_info_path).read())
        train_num = int(len(__data_info) * train_split)
        return __data_info[:train_num], __data_info[train_num:]


class DAIR2COCO(Loader):
    """DAIR-V2X数据集转换为coco格式"""

    def __init__(self, data_dir: str, train_split: float = 0.85) -> None:
        super(DAIR2COCO, self).__init__(data_dir, train_split)
        self.save_dir = os.path.join(data_dir, "pp-vehicle-annotations")

    @staticmethod
    def __get_annotations(annos_dir: str) -> dict:
        """获取标注信息"""
        return json.loads(open(annos_dir).read())

    @staticmethod
    def __bbox2xywh(bbox: dict) -> list:
        """将DAIR-V2X中的bbox[xmin, ymin, xmax, ymax], 转为coco的bbox[xmin, ymin, width, height]

        Args:
            bbox (dict): DAIR-V2X的bbox

        Returns:
            list: coco的bbox
        """
        return [
            float(bbox["xmin"]),
            float(bbox["ymin"]),
            float(bbox["xmax"]) - float(bbox["xmin"]),
            float(bbox["ymax"]) - float(bbox["ymin"]),
        ]

    def format2coco(self, data_info: dict, json_path: str) -> None:
        """转为COCO格式

        Args:
            ids (list): 图片ids
            json_path (str): annotations json 保存路径
        """
        item_id = 0
        coco_json = {"images": [], "annotations": [], "categories": []}
        for _info in tqdm(data_info):
            file_name = _info["image_path"]
            img_id, _ = os.path.splitext(os.path.basename(file_name))
            coco_json["images"].append({
                "id": int(img_id),
                "file_name": file_name,
                "width": 1920,
                "height": 1080,
            })

            annos = self.__get_annotations(
                os.path.join(self.data_dir,
                             _info["label_camera_std_path"]))  # 获取标注信息
            for _anno in annos:
                if (category := _anno["type"].lower()
                        in self.vhicles):  # 将所有的车辆类别都设为vehicle
                    category = "vehicle"
                if category_id := self.categories.get(
                        category) is not None:  # 获取类别id
                    xywh = self.__bbox2xywh(
                        _anno["2d_box"]
                    )  # coco的bbox[xmin, ymin, width, height]
                    coco_json["annotations"].append({
                        "id":
                        len(coco_json["annotations"]),
                        "image_id":
                        int(img_id),
                        "category_id":
                        category_id,
                        "bbox":
                        xywh,
                        "area":
                        xywh[-2] * xywh[-1],
                        "iscrowd":
                        0,
                    })
                    item_id += 1
        coco_json["categories"] = [{
            "id": _id,
            "name": _name
        } for _name, _id in self.categories.items()]  # 类别信息
        json.dump(
            coco_json,
            open(json_path, "w+", encoding="utf-8"),
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
        )  # 保存json

    def processing(self) -> None:
        """处理进程"""
        os.makedirs(self.save_dir, exist_ok=True)
        # 创建进程
        train_process = Process(
            target=self.format2coco,
            kwargs={
                "data_info": self.train_info,
                "json_path": os.path.join(self.save_dir, "train2017.json"),
            },
        )
        val_process = Process(
            target=self.format2coco,
            kwargs={
                "data_info": self.val_info,
                "json_path": os.path.join(self.save_dir, "val2017.json"),
            },
        )
        # 启动进程
        train_process.start()
        val_process.start()


def parse_args():
    parser = argparse.ArgumentParser(
        description="DAIR-V2X dataset to PP-vehicle format.")
    parser.add_argument("--data_dir", type=str, help="数据位置")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DAIR2COCO(args.data_dir).processing()
