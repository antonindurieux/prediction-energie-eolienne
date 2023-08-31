# Modélisation de l'énergie éolienne produite par région en France

Ce repository contient un notebook présentant l'analyse et la modélisation de l'énergie éolienne produite en France métropolitaine, en fonction des données météorologiques.  

## Données

Les données utilisées sont les jeux de données publiques suivant :

- [Puissance éolienne produite région par région depuis 2013](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def)  
- [Evolution de la puissance éolienne installée par région depuis 2001](https://opendata.reseaux-energies.fr/explore/?q=parc+annuel+de+production+%C3%A9olien+et+solaire&disjunctive.theme&disjunctive.publisher&disjunctive.maille-geographique&disjunctive.frequence-de-mise-a-jour&disjunctive.pas-temporel&sort=explore.popularity_score)  
- [Données météo par région depuis 2010](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm)

## Usage

1. Installez les packages nécessaires au sein de votre environnement Python : `pip install -r requirements.txt`.
2. Lancez le notebook `prediction_energy_eolienne.ipynb`. Les données seront téléchargées et sauvegardées automatiquement au premier lancement du notebook **mais cela peut prendre plus de 30 min**. Alternativement, si vous possédez déjà les fichiers csv en local, placez les directement dans un répertoire `data` créé à la racine du repository pour éviter de les télécharger à nouveau.  
3. En dehors du téléchargement des données, l'exécution du notebook devrait prendre uniquement quelques minutes.  
4. Sinon, le notebook exécuté est également disponible en html dans le fichier `prediction_energy_eolienne.html`, mais l'une des figures est interactive et ne sera pas visualisable sous ce format.
