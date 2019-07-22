import json
import math
import numpy as np
import pandas as pd


data = json.load(open('./instances_train2014.json', 'r'))
data1 = json.load(open('./instances_val2014.json', 'r'))
unseen_category = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'tennis racket', 'suitcase', 'zebra']
unseen_id = {} #key:category_id value:category_name
seen_id = {}
for cate in data['categories']:
    if cate['name'] in unseen_category:
        unseen_id[cate['id']] = cate['name']
    else:
        seen_id[cate['id']] = cate['name']

unseen_image_id = [] # key: image_id value: bbox
seen_image_id = []
unseen_category_id = []
seen_category_id = []
unseen_bbox = []
seen_bbox = []
unseen_image_path = []
seen_image_path = []
unseen_array = []
seen_array = []
for ann in data['annotations']:
    if ann['category_id'] in unseen_id:
        unseen_image_id.append(ann['image_id'])
        unseen_bbox.append(ann['bbox'])
        print("bbox:{}".format(ann['bbox']))
        unseen_category_id.append(unseen_id[ann['category_id']])
        print("ann:{}".format(unseen_id[ann['category_id']]))
    else:
        seen_image_id.append(ann['image_id'])
        seen_bbox.append(ann['bbox'])
        seen_category_id.append(seen_id[ann['category_id']])

for ann1 in data1['annotations']:
    if ann1['category_id'] in unseen_id:
        unseen_image_id.append(ann1['image_id'])
        unseen_bbox.append(ann1['bbox'])
        unseen_category_id.append(unseen_id[ann1['category_id']])
    else:
        seen_image_id.append(ann1['image_id'])
        seen_bbox.append(ann1['bbox'])
        seen_category_id.append(seen_id[ann1['category_id']])

id2path = {}
for img in data['images']:
    id2path[img['id']] = 'train2014/' + img['file_name']

for img1 in data1['images']:
    id2path[img1['id']] = 'val2014/' + img1['file_name']

for id in unseen_image_id:
    unseen_image_path.append(id2path[id])

for id_seen in seen_image_id:
    seen_image_path.append(id2path[id_seen])
unseen_bbox_int = []
print("unseen:{}".format(len(unseen_bbox)))
seen_bbox_int = []
for unseen in unseen_bbox:
    unseen_bbox_int.append(map(math.ceil, unseen))
for seen in seen_bbox:
    seen_bbox_int.append(map(math.ceil, seen))
print("unseen_int:{}".format(unseen_bbox_int))
a = np.array([unseen_image_path]).reshape(-1, 1)
print(a)
b = np.array([unseen_bbox]).reshape(-1, 4)
print(b)
c = np.array([[unseen_category_id]]).reshape(-1, 1)
print(c)
np_unseen = np.concatenate((a, b, c), axis=1)
d = np.array([seen_image_path]).reshape(-1, 1)
e = np.array([seen_bbox]).reshape(-1, 4)
f = np.array([[seen_category_id]]).reshape((-1, 1))
# print('d:{}\ne:{}\nf:{}\n'.format(d.shape, e.shape, f.shape))
np_seen = np.concatenate((d, e, f), axis=1)
csv_unseen = pd.DataFrame(np_unseen)
csv_seen = pd.DataFrame(np_seen)
csv_unseen.to_csv('./unseen_test.csv', index=False, header=False)
csv_seen.to_csv('./seen_test.csv', index=False, header=False)

