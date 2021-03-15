# Mode d'emploi

## But 
Identifier les étudiants su master SISE et leurs caractéristiques (comme l'âge, leur genre et leur expression). 

## Installation
Tout d'abord, créer un environnement pour éviter les conflits, et installer les packages suivants (avec Anaconda prompt):

```Bash
pip install opencv-contrib-python
pip install cmake
conda install -c conda-forge dlib
pip install face_recognition
pip install imutils
pip install keras
pip install streamlit
```
## Imports 
Dans un script Python, vérifier aussi que les imports suivant fonctionnent correctement :

```Python
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import math
import argparse
from pathlib import Path
import os
import ntpath
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import streamlit as st
import tempfile
```
## Démarrage

Une fois installés, rendez-vous dans le dossier de l'application. Nous avons 2 versions pour lancer la reconnaissance faciale. Pensez à modifier les chemins absolus pour qu'ils correspondent à l'emplacement de votre dossier. Les 2 versions se différencient simplement par le mode de lancement du traitement de vidéos :

_ face_recognition_app qu'il faudra run dans un terminal (après avoir changé de repertoire courant) via la commande ```Bash streamlit run face_recognition_app.py ``` qui permet de séléctionner une vidéo dans un repertoire à travers une application et de la visualiser directement dans celle-ci.

_ face_recognition_args qui prend en argument le chemin absolu vers un repertoire de videos avec l'option -v ou --video. Exemple dans Spyder :
```Python runfile('C:/Users/rdecoster/Downloads/BAROU_DECOSTER_PAVOINE/face_recognition_args.py', wdir='C:/Users/rdecoster/Downloads/BAROU_DECOSTER_PAVOINE', args='-v C:/Users/rdecoster/Downloads/PHOTOS+VIDEOS_SISE')``` et traitera les videos au format mp4 les unes à la suite des autres dans une fenêtre pop-up.

Dans les 2 cas, la ou les vidéos seront sauvegardées dans le répertoire courant.
