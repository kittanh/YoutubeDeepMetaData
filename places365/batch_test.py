# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, TensorDataset
import glob
import time
import datetime
import json 
import multiprocessing

start = time.time()

 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_labels():

    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv, weight_softmax, class_idx):

    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnTF():

# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():

    # this model has a last conv feature map as 14x14
    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


########### CHARGEMENT DE PARAMETRES ###########


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

################### PLAYLIST VIDEOS ###################

video_dir = 'videos/'

# list all files in the directory
all_files = os.listdir(video_dir)

# filter by video files
video_files = [f for f in all_files if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:

    ################### DECOUPAGE VIDEO ###################
    start2 = time.time()
    # création du dossier et sous dossier img

    dossier= "img"
    sous_dossier= "img"

    chemin_sous_dossier= os.path.join(dossier, sous_dossier)
    if not os.path.exists(chemin_sous_dossier):
        os.mkdir(dossier)
        os.mkdir(chemin_sous_dossier)
    else : 
        image_files = glob.glob(os.path.join(chemin_sous_dossier, '*.jpg'))
        # Use a loop to remove each file
        for file in image_files:
            os.remove(file)

    # ouvrir la vidéo
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))

    # récupérer le nombre total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # récupérer le délai entre les frames
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)))

    # on découpe toutes les 1/divisions secondes
    division = 1

    # initialiser le compteur de frames
    count = 0
    list_frames = []
    # boucle sur les frames
    while cap.isOpened():
        # lire le frame suivant
        ret, frame = cap.read()

        # sortir de la boucle si on atteint la fin de la vidéo
        if not ret:
            break

        # incrémenter le compteur de frames
        count += 1

        # sauvegarder le frame s'il est inclus dans l'intervalle
        if count % (fps // division) == 0:
            cv2.imwrite("img/img/frame_{}.jpg".format(count // fps), frame)
            list_frames.append(frame)  

    # libérer la vidéo
    cap.release()
    end2 = time.time()
    print("Temps d'exécution img : {:.2f} secondes".format(end2 - start2))
    ########### CREATION DATALOADER AND BATCH ###########

    folder_path = 'img'
    image_dataset = datasets.ImageFolder(root=folder_path, transform=tf)

    # Définir la taille du batch

    batch_size = 128
    count_frame = 1
    timestamp = 1

    # Créer un DataLoader pour charger les images en tant que batchs
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, num_workers = 1)

    list_1_video = []
    dict_1_video = {}
    dict_1_video["idVideo"] = video_file
    # forward pass sur chaque batch d'images
    start3 = time.time()
    for batch_idx, (data, target) in enumerate(image_loader):
        # CHARGEMENT DE L'IMAGE
        input_img = data
        print(np.shape(input_img))
        print(type(input_img))
        # forward pass sur le batch d'images
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(1, True)
        probs = probs.numpy()
        idx = idx.numpy()
        
        # affichage des résultats pour le batch en cours
        # print(f"BATCH {batch_idx} traité. Nombre d'images dans le batch : {len(data)}.")


        ########## OUTPUT ###########


        # print('RESULT ON BATCH ')

        # output the IO prediction
        # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
        # if io_image < 0.5:
        #     print('\n --TYPE OF ENVIRONMENT: indoor')
        # else:
        #     print('\n--TYPE OF ENVIRONMENT: outdoor')

        ########### SCENE CATEGORIES ###########

        # output the prediction of scene category
        # print('\n--SCENE CATEGORIES:')
        for j in range(batch_size):
            try :  
                scene_categories_dict = {}
                for i in range(365):
                    scene_categories_dict[classes[idx[j,i]]] = float(probs[j,i])
                # ajouter le dictionnaire pour cette image à la liste
                dict_1_frame = {}
                dict_1_frame['frame'] = (count_frame+j)
                dict_1_frame['timestamps'] = (str(datetime.timedelta(seconds=count_frame+j))) 
                dict_1_frame["scene_attribute"] = scene_categories_dict 
                list_1_video.append(dict_1_frame) 
            except IndexError:
                break 
        
        dict_1_video['features']=(list_1_video)

        count_frame+=batch_size
    end3 = time.time()
    print("Temps d'exécution_batch : {:.2f} secondes".format(end3 - start3))
    ################ JSON FILE ######################

    # convert list_scene_categories to JSON string
    json_str = json.dumps(dict_1_video)

    # save the JSON string to file
    with open(f'dict_{video_file}.json', 'w') as f:
        f.write(json_str)

end = time.time()
print("Temps d'exécution : {:.2f} secondes".format(end - start))