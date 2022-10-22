"""
@File    :   coco.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/09/01 18:08:09
@Desc    :   将TT-100K数据集转换为coco格式
"""

import argparse
import json
import os
from multiprocessing import Process
from typing import Tuple

from tqdm import tqdm


class Loader:

    def __init__(self, data_dir: str) -> None:
        """初始化

        Args:
            data_dir (str): TT-100K 数据集根目录
        """
        self.__data_dir = data_dir
        self.__annos16_dir = os.path.join(data_dir,
                                          "annotations.json")  # 2016 版标注信息
        self.__annos21_dir = os.path.join(data_dir,
                                          "annotations_all.json")  # 2021 版标注信息

        self.__categories, self.__annos = self.__get_annotations()
        self.__train_ids, self.__val_ids, self.__test_ids = self.__get_ids()

    @property
    def categories(self) -> dict:
        """类别与id对应关系"""
        return self.__categories

    @property
    def annotations(self) -> dict:
        """合并后的标注信息 (2021 + 2016)"""
        return self.__annos

    @property
    def train_ids(self) -> list:
        """训练集图片id"""
        return self.__train_ids

    @property
    def val_ids(self) -> list:
        """验证集图片id"""
        return self.__val_ids

    @property
    def test_ids(self) -> list:
        """获取测试集图片id"""
        return self.__test_ids

    def __get_annotations(self) -> Tuple[dict, dict]:
        """获取合并后的类别信息与标注信息"""
        __annos16 = json.loads(open(self.__annos16_dir).read())
        __annos21 = json.loads(open(self.__annos21_dir).read())
        __categories = sorted(
            list(set(__annos16["types"] + __annos21["types"])))  # 类别信息合并并排序

        return {
            category: category_id
            for category_id, category in enumerate(__categories)
        }, __annos16["imgs"] | __annos21["imgs"]

    def __get_ids(self) -> Tuple[list, list, list]:
        """获取图片id"""
        __train_path = os.path.join(self.__data_dir, "train/ids.txt")
        __val_path = os.path.join(self.__data_dir, "test/ids.txt")
        __test_path = os.path.join(self.__data_dir, "other/ids.txt")

        return (
            open(__train_path).read().splitlines(),
            open(__val_path).read().splitlines(),
            open(__test_path).read().splitlines(),
        )


class TT100k2COCO(Loader):

    def __init__(self, data_dir: str) -> None:
        super(TT100k2COCO, self).__init__(data_dir)
        self.save_dir = os.path.join(data_dir, "annotations")

    def __bbox2xywh(self, bbox: dict) -> list:
        """将TT-100K中的bbox[xmin, ymin, xmax, ymax], 转为coco的bbox[xmin, ymin, width, height]

        Args:
            bbox (dict): TT-100K的bbox

        Returns:
            list: coco的bbox
        """
        return [
            bbox["xmin"],
            bbox["ymin"],
            bbox["xmax"] - bbox["xmin"],
            bbox["ymax"] - bbox["ymin"],
        ]

    def format2coco(self, ids: list, json_path: str) -> None:
        """转为COCO格式

        Args:
            ids (list): 图片ids
            json_path (str): annotations json 保存路径
        """
        coco_json = {"images": [], "annotations": [], "categories": []}
        for item_id, image_id in enumerate(tqdm(ids)):
            anno = self.annotations[image_id]

            image_dict = {
                "file_name": anno["path"],
                "height": 2048,
                "width": 2048,
                "id": anno["id"],
            }

            coco_json["images"].append(image_dict)
            for item in anno["objects"]:
                xywh = self.__bbox2xywh(item["bbox"])
                category = item["category"]
                category_id = self.categories[category]
                annotation_dict = {
                    "area": xywh[-2] * xywh[-1],
                    "iscrowd": 0,
                    "image_id": anno["id"],
                    "bbox": xywh,
                    "category_id": category_id,
                    "id": item_id,
                }

                coco_json["annotations"].append(annotation_dict)
                if category not in coco_json["categories"]:
                    coco_json["categories"].append(category)
        categories_list = [{
            "id": self.categories[category],
            "name": category
        } for category in coco_json["categories"]]

        coco_json["categories"] = categories_list
        with open(json_path, "w+", encoding="utf-8") as file:
            json.dump(coco_json,
                      file,
                      indent=4,
                      sort_keys=False,
                      ensure_ascii=False)

    def processing(self) -> None:
        """处理进程"""
        os.makedirs(self.save_dir, exist_ok=True)
        # 创建进程
        train_process = Process(
            target=self.format2coco,
            kwargs={
                "ids": self.train_ids,
                "json_path": os.path.join(self.save_dir, "train2017.json"),
            },
        )
        val_process = Process(
            target=self.format2coco,
            kwargs={
                "ids": self.val_ids,
                "json_path": os.path.join(self.save_dir, "val2017.json"),
            },
        )
        test_process = Process(
            target=self.format2coco,
            kwargs={
                "ids": self.test_ids,
                "json_path": os.path.join(self.save_dir, "test2017.json"),
            },
        )
        # 启动进程
        train_process.start()
        val_process.start()
        test_process.start()


def parse_args():
    parser = argparse.ArgumentParser(
        description="TT-100K dataset to COCO format.")
    parser.add_argument("--data_dir", type=str, help="数据位置")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TT100k2COCO(args.data_dir).processing()
