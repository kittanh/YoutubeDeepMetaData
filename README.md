# YoutubeDeepMetaData

## Comment exécuter le projet ?

### Installation des packages

Avant d'exécuter tout code, il est nécessaire d'installer les packages présents dasn le fichier requirements.txt.
Pour ce faire, se placer à la racine du projet. Ensuite, faire la commande :

```shell
pip install requirements.txt
```

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
## Et après ?

Il est possible de consulter et faire des requêtes sur les données générées. Pour cela, voir le dossier "flask".
