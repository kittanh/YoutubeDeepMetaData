# Contexte

Il est possible de visualiser les données générées et de faire des requêtes dessus.

# Flask

## Initialisation de l'environnement

1) Lancer elasticsearch au préalable. Pour nos essais, l'image utilisée sur Docker est docker.elastic.co/elasticsearch/elasticsearch:7.4.0

2) Lancer le programme main.py contenu dans le dossier flask. Le programme met beaucoup de temps à s'exécuter (environ 5 minutes selon la puissance de l'ordinateur et le nombre de fichiers JSON générés).

3) Une fois le programme lancé, cliquer sur le lien: http://127.0.0.1:5000/ pour accéder à l'application Flask. 

## Utilisation de l'application

Sur la page d'accueil, on peut observer les données pour les 1000 premières vidéos. On peut alors utiliser les différentes fonctionnalités.

# Requêtes

Nous pouvons effectuer des requêtes sur deux champs : le lieu et l'objet. Nous avons créé plusieurs fonctions :

La fonction ‘generate_data()’ prend en argument un fichier json généré par les algorithmes YOLO et PLACES 365 et et l’insère dans l’index video.

La fonction ‘search_lieu(query)’ permet de rechercher les vidéos contenant un lieu passé en paramètre. Puis, une requête Elasticsearch est utilisée pour filtrer les vidéos en fonction du lieu spécifié. Un filtrage est ensuite effectué pour que la fonction renvoie chaque frame de chaque vidéo qui contient le lieu spécifique. 

La fonction ‘search_object(var_object)’ permet de rechercher les vidéos contenant un objet passé en paramètre. Le principe est le même que la fonction précédente ‘search_lieu(query)’, la fonction renvoie seulement les frames des vidéos qui contient le lieu spécifique.
