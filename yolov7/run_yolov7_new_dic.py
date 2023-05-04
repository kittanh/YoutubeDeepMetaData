import argparse
import time
from pathlib import Path

import numpy as np

import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
# from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample,
#     UniformCropVideo
# )
from typing import Dict

classes_to_filter = ['train'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt  = {
    
    "weights": "weights/yolov7.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '1',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter,  # list of classes to filter or None
    "single_cls" : False

}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def run_yolov7_videos(paths, batch_size, division):
  videos = []
  for video_path in paths:
    vid = dict()
    title = video_path.split('/')[-1]
    vid['idVideo'] = title
    feat, fps = run_yolov7_batch(video_path, 32, division)
    vid['fps'] = fps
    vid['features'] = feat
  return videos

### sur plusieurs vidéos
def run_yolov7_batch(video_paths, batch_size, division):
  torch.cuda.empty_cache()
  
  # Initializing model and setting it for inference
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    
    # Initialize
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
  

    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names

    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:
        classes.append(names.index(class_name))

    if classes:
      classes = [i for i in range(len(names)) if i not in classes]

    cudnn.benchmark = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # videos = dict()

    videos = []
    
    

    
    num = 6
    json_nom = 'all_vid_yolo_' + str(num) + '.json'

    for num_video, video_path in enumerate(video_paths):
      if num_video >= 5363:
        if num_video%1000 == 0 and num_video!=0:
          num += 1
          json_nom = 'all_vid_yolo_' + str(num) + '.json'

        if num_video == 0 or num_video%1000== 0:
          with open(json_nom, 'a') as f:
            f.write('[')



        vid = dict()
        title = video_path.split('/')[-1]
        print(num_video, title)
        vid['idVideo'] = title

        


        video = cv2.VideoCapture(video_path)
        nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)
        # ret, img0 = video.read()
        fps = round(float(video.get(cv2.CAP_PROP_FPS)))
        if nframes <= fps:
          division = fps
        else:
          division = 1

          fps = fps // division

          nframes = nframes//fps
        # video.set(cv2.CAP_PROP_FPS, fps) # set FPS
            # duration = nframes / fps * 1000

        

        vid['fps'] = fps

        dataset = LoadImages(video_path, img_size=imgsz, stride=stride)
        
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        batch = []
        # nframes = dataset.nframes
        count_frame = 0
        n_batch = nframes//batch_size
        count_batch = 0
        reste = nframes%batch_size

        

        t0 = time.time()
        obj = []

        count_frame_reel = 0
        count_frame_4_batch = 0
        for path, img, im0s, vid_cap in dataset:
          if count_frame_reel % (fps // division) == 0:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            batch.append(img)
            count_frame+=1
          # print(count_frame)
          count_frame_reel+=1

          # Inference
          t1 = time_synchronized()

          if count_frame == batch_size or (count_batch == n_batch and count_frame == reste and reste != 0):

            images = torch.stack(batch)
            
            #images = images.flatten(start_dim=5, end_dim=4)
            b, n, c, h, w = images.shape
            images = images.reshape(b, n*c, h, w)

            pred = model(images, augment= False)[0]
            

            t2 = time_synchronized()


            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)

            # count_batch+=1
            # batch = []

            for i, det in enumerate(pred):
              # p, im0, frame = path[i], im0s[i].copy(), dataset.count
              # s=""
              labels = dict()
              
              count_frame_4_batch+=1
              # if (count_batch == n_batch and count_frame == reste):
              #   nfr = count_frame - reste + i + 1
              # else:
              #   nfr = count_frame - batch_size + i + 1

              labels["frame"] = count_frame_4_batch
              # labels["timestamp"] = str(datetime.timedelta(milliseconds=int(count_frame * duration / nframes)))
              
              # objects = dict()#création du dictionnaire des positions des objets 
              labels['objects'] = []
              labels['conf'] = []
              labels['bndbox'] = []
              gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
                
              for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() #normalise les données de la position des objets 
                name = names[int(cls)]
                # if name not in objects: 
                #   objects[name] = []
                # objects[name].append(dict())
                # objects[name][-1]['bndbox'] = dict()
                # objects[name][-1]['bndbox']['xmin'] = xywh[0]
                # objects[name][-1]['bndbox']['xmax'] = xywh[1]
                # objects[name][-1]['bndbox']['ymin'] = xywh[2]
                # objects[name][-1]['bndbox']['ymax'] = xywh[3]
                # objects[name][-1]['conf'] = conf.item()
                labels['objects'].append(name)
                labels['conf'].append(conf.item())
                labels['bndbox'].append(xywh)

              
              # labels["objects"] = objects
              obj.append(labels)
            # print(count_batch)
            count_batch+=1
            batch = []
            count_frame = 0
      # videos[title] = obj
        vid['features'] = obj
        # print(len(obj))
        # videos.append(vid)
      
        json_str = json.dumps(vid)

        # save the JSON string to file
        last_char = ', '
        if num_video == len(video_paths)-1 or num_video%1000==999:
          last_char = "]"

        with open(json_nom, 'a') as f:
            f.write(json_str + last_char)

    t3 = time_synchronized()
    print(len(obj))
    print(f'Done. ({time.time() - t0:.3f}s)')
    return(videos)#, fps


    with open('all_vid_yolo.json', 'a') as f:
      f.write(']')

# def load_yolo_model():
#   weights, imgsz = opt['weights'], opt['img-size']
  
#   # Initialize
#   set_logging()
#   device = select_device(opt['device'])
#   half = device.type != 'cpu'

#   # Load model
#   model = attempt_load(weights, map_location=device)  # load FP32 model
#   stride = int(model.stride.max())  # model stride
#   imgsz = check_img_size(imgsz, s=stride)  # check img_size


#   if half:
#     model.half()

#   names = model.module.names if hasattr(model, 'module') else model.names

#   classes = None
#   if opt['classes']:
#     classes = []
#     for class_name in opt['classes']:
#       classes.append(names.index(class_name))

#   if classes:
#     classes = [i for i in range(len(names)) if i not in classes]

#   cudnn.benchmark = True

#   # Get names and colors
#   names = model.module.names if hasattr(model, 'module') else model.names
#   colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#   return [imgsz, stride, device, model, half, classes, names]

# def run_yolov7_batch(video_path, batch_size, division, param):
#   title, imgsz, stride, device, model, half, classes, names = param

#   # videos = []

#   vid = dict()
#   title = video_path.split('/')[-1]
#   print(title)
#   vid['idVideo'] = title

  
#   video = cv2.VideoCapture(video_path)
#   # ret, img0 = video.read()
#   fps = round(float(video.get(cv2.CAP_PROP_FPS)))
#   fps = fps // division
#   # video.set(cv2.CAP_PROP_FPS, fps) # set FPS
#   nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)//fps # duration = nframes / fps * 1000

#   vid['fps'] = fps

#   dataset = LoadImages(video_path, img_size=imgsz, stride=stride)
  
#   if device.type != 'cpu':
#       model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

#   batch = []
#   # nframes = dataset.nframes
#   count_frame = 0
#   n_batch = nframes//batch_size
#   count_batch = 0
#   reste = nframes%batch_size

#   t0 = time.time()
#   obj = []

#   count_frame_reel = 0
#   count_frame_4_batch = 0
#   for path, img, im0s, vid_cap in dataset:
#     if count_frame_reel % (fps // division) == 0:
#       img = torch.from_numpy(img).to(device)
#       img = img.half() if half else img.float()  # uint8 to fp16/32
#       img /= 255.0  # 0 - 255 to 0.0 - 1.0
#       if img.ndimension() == 3:
#           img = img.unsqueeze(0)

#       batch.append(img)
#       count_frame+=1
#     # print(count_frame)
#     count_frame_reel+=1

#     # Inference
#     t1 = time_synchronized()

#     if count_frame == batch_size or (count_batch == n_batch and count_frame == reste and reste != 0):

#       images = torch.stack(batch)
      
#       #images = images.flatten(start_dim=5, end_dim=4)
#       b, n, c, h, w = images.shape
#       images = images.reshape(b, n*c, h, w)

#       pred = model(images, augment= False)[0]
      

#       t2 = time_synchronized()


#       pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)

#       # count_batch+=1
#       # batch = []

#       for i, det in enumerate(pred):
#         # p, im0, frame = path[i], im0s[i].copy(), dataset.count
#         # s=""
#         labels = dict()
        
#         count_frame_4_batch+=1
#         # if (count_batch == n_batch and count_frame == reste):
#         #   nfr = count_frame - reste + i + 1
#         # else:
#         #   nfr = count_frame - batch_size + i + 1

#         labels["frame"] = count_frame_4_batch
#         # labels["timestamp"] = str(datetime.timedelta(milliseconds=int(count_frame * duration / nframes)))
        
#         # objects = dict()#création du dictionnaire des positions des objets 
#         labels['objects'] = []
#         labels['conf'] = []
#         labels['bndbox'] = []
#         gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
          
#         for *xyxy, conf, cls in reversed(det):
#           xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() #normalise les données de la position des objets 
#           name = names[int(cls)]
#           # if name not in objects: 
#           #   objects[name] = []
#           # objects[name].append(dict())
#           # objects[name][-1]['bndbox'] = dict()
#           # objects[name][-1]['bndbox']['xmin'] = xywh[0]
#           # objects[name][-1]['bndbox']['xmax'] = xywh[1]
#           # objects[name][-1]['bndbox']['ymin'] = xywh[2]
#           # objects[name][-1]['bndbox']['ymax'] = xywh[3]
#           # objects[name][-1]['conf'] = conf.item()
#           labels['objects'].append(name)
#           labels['conf'].append(conf.item())
#           labels['bndbox'].append(xywh)

        
#         # labels["objects"] = objects
#         obj.append(labels)
#       # print(count_batch)
#       count_batch+=1
#       batch = []
#       count_frame = 0
# # videos[title] = obj
#   vid['features'] = obj
#   # print(len(obj))
#   # videos.append(vid)

#   json_str = json.dumps(vid)

#   # save the JSON string to file
#   with open('testtt.json', 'a') as f:
#       # f.write(json_str)
#       # f.seek(0, 2)
#       # if f.tell() > 0:
#       f.write(json_str)

#   # print(len(obj))
#   # print(f'Done. ({time.time() - t0:.3f}s)')
#   # return(videos)#, fps

def main():

  # video_dir = "/home/deepmetadata/places365/videos/"
  video_dir = "/storage8To/datasets/APY_YOUTUBE_DATASET/videos/"

  # list all files in the directory
  all_files = os.listdir(video_dir)
  video_files = [video_dir+f for f in all_files if f.endswith(('.mp4', '.avi', '.mov'))]
  # video_files = ["/home/deepmetadata/places365-1/videos/Inside Lenny Kravitz's Brazilian Farm Compound _ Open Door _ Architectural Digest-FlsKjWqu82k.mp4"]
  # print(video_files)
  # res = run_yolov7_batch(video_path, 32, 1)
  # print(len(res[0]))
  # print(res)
  # for r in res:
  #    print(r, '\n')s

  # video_files = ["/home/deepmetadata/yolov7/Test.mp4"]
  # video_files = ["/home/deepmetadata/places365-1/videos/Ficello - formes 'le fromage trop rigolo' Pub 30s-qOQk5_aQfD4.mp4"]
  # res = run_yolov7_batch(video_files, 32, 1)
  run_yolov7_batch(video_files, 128, 1)

  # json_str = json.dumps(res)

    # save the JSON string to file
  # with open(f'new_dict_all_vid_yolo.json', 'w') as f:
  # # with open(f'dict_ficello.json', 'w') as f:
  #     f.write(json_str)
        
  # for vid in res:
  #    print(vid, '\n')

if __name__ == "__main__":
    main()

