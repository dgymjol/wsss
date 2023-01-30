import os
import numpy as np
import argparse
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio.v2 as imageio

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Computer Vision Task Training Processor')

    parser.add_argument(
        '--root',
        default='./data/VOCdevkit/VOC2012')
    
    return parser

parser = get_parser()
arg = parser.parse_args()
    
file_list = os.path.join(arg.root, "ImageSets/SegmentationAug/train_aug_pseudo" + ".txt")
file_list = tuple(open(file_list, "r"))
file_list = [id_.rstrip().split(" ") for id_ in file_list]
files, gt_labels = list(zip(*file_list))
dataset = VOCSemanticSegmentationDataset(split='train', data_dir=arg.root)
labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

preds = []
for id in dataset.ids:
    cls_labels = imageio.imread(os.path.join(arg.root, 'SegmentationPseudoClassAug', id + ".png")).astype(np.uint8)
    cls_labels[cls_labels == 255] = 0
    preds.append(cls_labels.copy())
    # if id == '2007_000170':
    #     labels = labels[:len(preds)]
    #     break
    
confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]
gtj = confusion.sum(axis=1)
resj = confusion.sum(axis=0)
gtjresj = np.diag(confusion)
denominator = gtj + resj - gtjresj

# fp = 1. - gtj / denominator
# fn = 1. - resj / denominator
iou = gtjresj / denominator

# print(fp[0], fn[0])
# print(np.mean(fp[1:]), np.mean(fn[1:]))

print({'iou': iou, 'miou': np.nanmean(iou)})