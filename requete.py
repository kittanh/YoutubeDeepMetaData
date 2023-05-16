from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import time

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

data= json.load(open("all_vids_combined_tuple_1.json", encoding='utf-8'))
data+= json.load(open("all_vids_combined_tuple_2.json", encoding='utf-8'))
data+= json.load(open("all_vids_combined_tuple_3.json", encoding='utf-8'))
data+= json.load(open("all_vids_combined_tuple_4.json", encoding='utf-8'))
data+= json.load(open("all_vids_combined_tuple_5.json", encoding='utf-8'))

def generate_data(data):
    for video in data:
        features=[]
        for feat in video["features"]:
            lieu = [(str(x[0]), str(x[1])) for x in feat["lieu"]]
            objects= [(str(x[0]), str(x[1])) for x in feat["objects"] if feat["objects"] is not None]
            bndbox= [(x[0], x[1], x[2], x[3]) for x in feat["bndbox"]]
            features.append({"frame": feat["frame"], "objects": objects, "bndbox": bndbox, "lieu": lieu})
        yield {
            "_index": "video",
            "_type": "features",
            "_source": {
                "idVideo": video["idVideo"],
                "fps": video["fps"],
                "features": features
            }
        }

if es.indices.exists('video')==True:
    es.indices.delete(index='video')
    bulk(es, generate_data(data))
else :
    bulk(es, generate_data(data))

time.sleep(5)

def search_init():
    result = es.search(index="video", query= {"match_all": {}},size=1000)

    results = []
    [results.append(elt['_source']) for elt in result["hits"]["hits"]]

    return results

def search_lieu(query):
    print(query)
    QUERY = { "query": {
    "bool": {
    "must": [],
    "filter": [
        {
        "bool": {
            "should": [
            {
                "match_phrase": {
                "features.lieu": query
                }
            }
            ],
            "minimum_should_match": 1
        }
        }
    ],
    "should": [],
    "must_not": []
    }
    }
    }
    result = es.search(index = "video", body= QUERY, size=1000)
    videos = [elt['_source'] for elt in result["hits"]["hits"]]
    res = []
    for video in videos : 
        dic_res = {}
        dic_frames = {}
        for frame in video['features']:
            for lieu in frame['lieu']:
                if query in lieu[0]:
                    if video['idVideo'] in dic_res:
                        num_frame = frame['frame']
                        dic_frames[num_frame] = lieu
                    else :
                        dic_res['idVideo'] = video['idVideo']
                        num_frame = frame['frame']
                        dic_frames[num_frame] = lieu
                dic_res['frames'] = dic_frames
        res.append(dic_res)
    return res

def search_object(var_object):
    QUERY = {
    "query": {
    "bool": {
      "must": [],
      "filter": [
        {
          "bool": {
            "filter": [
              {
                "bool": {
                  "should": [
                    {
                      "match_phrase": {
                        "features.objects": var_object
                      }
                    }
                  ],
                  "minimum_should_match": 1
                }
              }
            ]
          }
        }
      ],
      "should": [],
      "must_not": []
    }
  }
}
    result = es.search(index = "video", body= QUERY, size=1000)
    videos = [elt['_source'] for elt in result["hits"]["hits"]]
    res = []
    for video in videos : 
        dic_res = {}
        dic_frames = {}
        for frame in video['features']:
            for object in frame['objects']:
                if var_object in object[0]:
                    if video['idVideo'] in dic_res:
                        num_frame = frame['frame']
                        dic_frames[num_frame] = object
                    else :
                        dic_res['idVideo'] = video['idVideo']
                        num_frame = frame['frame']
                        dic_frames[num_frame] = object
                dic_res['frames'] = dic_frames
        res.append(dic_res)
    return res

