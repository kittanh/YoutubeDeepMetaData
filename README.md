# YoutubeDeepMetaData

## Comment exécuter le projet ?
### Pour exécuter PLACES365
batch_playlist.py

### Pour exécuter YOLOv7
run_yolov7_new_dic.py
Il est possible de modifier les paramètes à la ligne 491 :
run_yolov7_batch(video_paths, batch_size, division, json_size, last_video)

### Si l'un des deux scripts s'arrête, comment reprendre là où cela s'est arrêté ?
On regarde dans le terminal le numéro de la dernière vidéo.
#### Pour YOLOv7
Modifier la ligne 491 : run_yolov7_batch(video_paths, batch_size, division, json_size, last_video)
Il faut remplacer l'argumenter last_video avec le numéro de la dernière vidéo
