# YoutubeDeepMetaData

## Comment exécuter le projet ?
### Pour exécuter PLACES365
```shell
places365/batch_playlist.py
```

### Pour exécuter YOLOv7
```shell
yolov7/run_yolov7_new_dic.py  
```
Il est possible de modifier les paramètes à la ligne 491 :
```python
run_yolov7_batch(video_paths, batch_size, division, json_size, last_video)
```

### Si l'un des deux scripts s'arrête, comment reprendre là où cela s'est arrêté ?
On regarde dans le terminal le numéro de la dernière vidéo.
#### Pour YOLOv7
Modifier la ligne 491 de 
```shell
yolov7/run_yolov7_new_dic.py  
```
: 
```python
run_yolov7_batch(video_paths, batch_size, division, json_size, last_video) 
```
Il faut remplacer l'argumenter last_video avec le numéro de la dernière vidéo
#### Pour PLACES365
Modifier dans 
```shell
places365/batch_playlist.py
```
à la ligne 180 :
```python
num = num_du_dernier_json 
```  
Et à la ligne 188 :
```python
if num_video >= num_de_la_derniere_video:
```  


### Pour combiner les JSON de PLACES365 et YOLOv7
```shell
main/combine_dic.py  
```
#### Choisir quels JSON combiner ?
Modifier la ligne 93 de 
```shell
main/combine_dic.py  
```
: 
```python
for i in range(num_premier_json, num_dernier_json):
```
## Comment utiliser les bases de données créées ?

Une fois les données générées, il est possible de faire des requêtes via une interface réalisée avec Flask.
Pour cela, il faut tout d'abord avoir lancé elasticsearch. Nous avons utilisé l'image suivante sur Docker : docker.elastic.co/elasticsearch/elasticsearch:7.4.0
