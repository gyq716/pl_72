import json
import pandas as pd 
import numpy as np 
import argparse
from pycocotools.coco import COCO



def main():
    dic_coco = json.load(open('dic_coco.json', 'r'))
    train_id = []
    for im in dic_coco['images']:
        if im['split'] == "train":
            train_id.append(im['id'])
        else:
            continue
    file_path_train = 'MSCOCO/instances_val2014.json'
    train2014 = COCO(file_path_train)
    print("imgToAnns:{}\n".format(train2014.imgToAnns)
    # ann = train2014.loadAnns(train_id)
    # ann_bbox = ann['']
    # print(val2014.keys())
    # print('annotations:{}'.format(val2014['annotations'][:10]))


if __name__ == '__main__':
    main()


