import json

def combine_json(json_file_p, json_file_y):
    with open(json_file_p, 'r') as f:
        lst_p = json.load(f)
    with open(json_file_y, 'r') as f:
        lst_y = json.load(f)


    for i in range(len(lst_p)):
        vid = lst_p[i]
        if vid['idVideo'] == lst_y[i]['idVideo']: #les vidéos doivent être dans le même ordre
            for j in range(len(vid['features'])):
                lst_y[i]['features'][j]['scene_attribute'] = vid['features'][j]['scene_attribute']
            # print(len(vid['features']))
            # print(len(lst_y[i]['features']))

    return lst_y

def combine_json_new_dic(json_file_p, json_file_y):
    with open(json_file_p, 'r') as f:
        lst_p = json.load(f)
    with open(json_file_y, 'r') as f:
        lst_y = json.load(f)


    for i in range(len(lst_p)):
        vid = lst_p[i]
        if vid['idVideo'] == lst_y[i]['idVideo']: #les vidéos doivent être dans le même ordre
            for j in range(len(vid['features'])): #boucle sur les frames en gros
                for key in ['lieu', 'probabilite']:
                    if key in vid['features'][j]:
                        lst_y[i]['features'][j][key] = vid['features'][j][key]

    return lst_y


def combine_json_new_dic(json_file_p, json_file_y):
    with open(json_file_p, 'r') as f:
        lst_p = json.load(f)
    with open(json_file_y, 'r') as f:
        lst_y = json.load(f)


    for i in range(len(lst_p)):
        vid = lst_p[i]
        if vid['idVideo'] == lst_y[i]['idVideo']: #les vidéos doivent être dans le même ordre
            for j in range(len(vid['features'])): #boucle sur les frames en gros
                for key in ['lieu', 'probabilite']:
                    if key in vid['features'][j]:
                        lst_y[i]['features'][j][key] = vid['features'][j][key]

    return lst_y

def make_tuple(json_file_p, json_file_y):
    dic = combine_json_new_dic(json_file_p, json_file_y)
    
    for vid in dic:
        
        new_features = []
        for frame in vid['features']:
            # print(frame)
            new_dic = dict()
            
            new_dic['frame']=frame['frame']
            new_dic['objects']=[]
            for i in range(len(frame['objects'])):
                new_dic['objects'].append(tuple((frame['objects'][i], frame['conf'][i])))
            new_dic['bndbox']=frame['bndbox']
            new_dic['lieu']=[]
            for j in range(len(frame['lieu'])):
                new_dic['lieu'].append(tuple((frame['lieu'][j], frame['probabilite'][j])))
            new_features.append(new_dic)
            # print(new_features)
        vid['features'] = new_features
    return dic


if __name__ == "__main__":
    # json_file_p="/home/deepmetadata/places365-1/all_videos.json"
    # json_file_y="/home/deepmetadata/yolov7/dict_all_vid_yolo.json"

    # res = combine_json(json_file_p, json_file_y)

    # json_str = json.dumps(res)

    # # save the JSON string to file
    # with open(f'dict_all_vid.json', 'w') as f:
    #     f.write(json_str)

    ### new dic
    for i in range(10, 24+1):
        json_file_p="/home/deepmetadata/places365/all_vid_places_"+str(i)+".json"
        json_file_y="/home/deepmetadata/yolov7/all_vid_yolo_"+str(i)+".json"

        # res = combine_json_new_dic(json_file_p, json_file_y)
        res = make_tuple(json_file_p, json_file_y)

        json_str = json.dumps(res)

        # save the JSON string to file
        with open("all_vids_combined_tuple_"+str(i)+".json", 'w') as f:
            f.write(json_str)
