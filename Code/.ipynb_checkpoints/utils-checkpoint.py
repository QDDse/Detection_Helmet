import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import sys
import numpy as np, scipy

## Convert str to int
def str2int(inp_list):
    return list(map(int, inp_list))

def Convert_label2xml(txt_path, target_path=None):
    '''
    将txt格式的label 转化成xml
    args:
        txt_path: Path of the txt
        target_path: Path to save new format
    '''
    label_dict = {}
    bboxes = []
    with open(txt_path, 'r') as F:
        rows = F.readlines()
        F.close()
    # print(rows)
    num = len(rows)
    head_id = []
    for i, row in enumerate(rows):
        row = row.strip('\n')
        # print(row)
        # print(row.split(' ')[0])
        if row.split(' ')[0] == 'head':
            head_id.append(i)
        bbox = row.split(' ')[1:]
        ## 转化为 int
        bboxes.append(str2int(bbox))
    # print(bboxes)
    # print(head_id)
    label_dict['boxes'] = torch.from_numpy(np.array(bboxes))
    labels = torch.ones(num) 
    labels[head_id] = 0
    label_dict['labels'] = labels
    return label_dict

# functions to display samples a/w bounding box

def img_show(img, ax = None, figsize=(16,8)): #function to show image
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.tick_top()
    ax.imshow(img)
    return ax
        
def draw_box(img, ax, bb, lbl): #func to draw bounding box
    if lbl=='helmet':
        color_lbl='red'
    elif lbl=='head':
        color_lbl='yellow'
    else:
        color_lbl='blue'
    
    # rectangle draws bbox around the class
    rect = patches.Rectangle(
        (int(bb[0]),int(bb[1])), int(bb[2])-int(bb[0]), int(bb[3])-int(bb[1]),
        fill=False, edgecolor=color_lbl, lw=2
    )
    ax.add_patch(rect)


def plot_sample(a_img, tgt, ax=None, figsize=(16,8)):
    img = np.array(a_img).copy() #making deep copy for reproducibility (else if run 2nd time it gives err)
    img = img*255 #multiplying with 255 becoz in augmtn pixels r normalized
    img = np.transpose(img, (2,1,0)) #making channel 1st to channel last (i.e from (3,416,416) to (416,416,3))
    tr_img = scipy.ndimage.rotate(img, 270, reshape=False) #rotatg becoz rcvd img was hztlly left faced
    tr_img = np.flip(tr_img, axis=1) #mirroring the image becoz rcvd img was flipped
    ax = img_show(tr_img, ax=ax)
    
    for box_id in range(len(tgt['boxes'])):
        box = tgt['boxes'][box_id] #target['boxes'][box_id] contains (xmin, ymin, xmax, ymax) i.e bbox coor for each label
        lbl = CLASS_NAME[tgt['labels'][box_id]] #converting index back to str labels i.e 1 to 'helmet'
        draw_box(tr_img, ax, box, lbl) #drawing multiple bbox on single image using matplotlib