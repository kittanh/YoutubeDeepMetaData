# Contexte

Il est possible de visualiser les données générées et de faire des requêtes dessus.

# Flask

## Initialisation de l'environnement

1) Lancer elasticsearch au préalable. Pour nos essais, l'image utilisée sur Docker est docker.elastic.co/elasticsearch/elasticsearch:7.4.0

2) Lancer le programme main.py contenu dans le dossier flask. Le programme met beaucoup de temps à s'exécuter (environ 5 minutes selon la puissance de l'ordinateur et le nombre de fichiers JSON générés).

3) Une fois le programme lancé, cliquer sur le lien: http://127.0.0.1:5000/ pour accéder à l'application Flask. 

## Utilisation de l'application

Sur la page d'accueil, on peut observer les données pour les 1000 premières vidéos. On peut alors utiliser les différentes fonctionnalités.
