# Internship Report
## Introduction
## Context, Methodology and Work Plan
- Objectif : Développement et optimisation d'un outil de segmentation des lésions cérébrales sur IRM après AVC 
- Modèle IA existant qui fait la segmentation avec nnUNET : description du modèle ...
- nnUNET est très gros très lourd, fait pour faire de l'entrainement, mais permet aussi l'inférence pure
- Packaging d'une app (pas web car donnée sensible etc...) : objectif être compatible sur plusieurs OS mais objectif principal Windows pour médecin et secondaire Linux (Fedora), MacOs pour chercheur
- Premier objectif était juste une app en command line mais après avoir montrer un exemple d'app graphique simple, médecin ravi donc double objectif : app graphique et command line
- Packaging app assez légère et complètement "autonome ?" : utilisable par des médecins donc doit embarquer toutes les dépendances etc ...
- Plusieurs possibilités de packaging : python embed sur windows, pyinstaller sur tous les OS, py2app etc ...
- app composée de : preprocessing (fait par les chercheurs sur les données avant entrainement par nnUNet + fait par nnUNet (tout est éparpillé)), inference (reprendre ce que fait nnUNet ? reprendre à 0 ? au début pyTorch mais changement au milieu), et postprocessing (même combat que preprocessing)
- Choix techno : python ? car largement utilisé pour faire de la recherche + simple + nnUNet en python et entrainer mais moyennement adapté pour packaging ? C/C++ packaging mais plus longue prise en main et inconnu dans le service ...
- Liste des tâches proposées par médecin + tuteurs : 
    - Afficher un Disclaimer : For research purpose only in the terminal/help command/in the result folder, README) (like HD-BET)
    - Segmentation sur template en choix (T1 /Flair/MNI, default T1 et FLAIR)
    - Sauvegarde des images individuelles du preprocessing (en option non activée par default, et sauvegarde selective des étapes de preprocess)
    - Process par batch / Paralléliser les traitements
    - Functionality DICOM et NIFTI (in the input)
    - BIDS Output
    - optional Sauvegarder pmap 
    - Segmentation (avec option threshold, par default 0.2/0.5)
    - Comparaison inference GPU/CPU
    - Afficheur up to user (le logiciel lance FSL/ITK/MedINRIA)
    - OS : Windows (PC portable, demander à la DSI)/Linux (GPU-Empenn)
    - Diffusion : Format package pip  / Voir SpinalCord Toolbox
    - Documentation a faire
    - Option pour charger un modèle externe (et le fichier config du modele)
    - Licence : AGPL ? (a demander à Francesca d’aller voir ça avec le STIP) 

## Proposed Solution / Work Performed / Results
dans l'ordre : 
### Documentation, découverte des technologies premiers tests externes à l'app elle même ...
- Documentation : Nifti, Dicom, BIDS, Pytorch, Deep Learning and Multi-Modal MRI for the Segmentation of Sub-Acute and Chronic Stroke Lesions
(papier scientifique sur le modèle developpé)
- Estimation des tâches : redéfinir précisemment les taches de la liste de tache, essayer de mettre une note (suite de fibonnaci comme en méthode agile) de difficultée, première réflexion sur la réalisation future
- Visualisation et étude d'image Nifti DIcom avec itksnap et python avec nibabel etc ... comprendre le format, j'ai fait un poc de convertisseur de Dicom en nifti (pour mieux comprendre les formats)
- Découverte de docstring, découverte de la convertion en documentation HTML à partir de docstring
- Test de packaging d'un module en package pip avec TestPyPI
- Test executable avec pyinstaller sur différents OS + py2app (pas encore de machine windows)
- découverte de dcm2niix
- Premier POC pour display image en fonction d'un fichier .ini qui conserve visualiseurs par défaut, chemin d'executable etc
- Test de construction d'un modèle ultra léger pytorch (jamais utilisé, tensorflow en cours), et faire inférence avec
- Découverte de libtorch (C++)
- Découverte et test de nnUNet, prédiction sur plusieurs machines et environnement (récupération d'un pc windows)
- Première app à partir d'un nnUNet nettoyé pour prediction uniquement (comparaison taille app et venv en fonction de CUDA ou pas)
- Test d'une app graphique, ça a convaincu l'équipe de faire à la fois une app command line et graphique simple avec tkinter
### Packaging et developpement de l'app
- Elaboration d'une architecture pour l'application finale (preproc, inference, postproc)
- Développement de l'app ultra simple avec preproc, l'inférence en pytorch et postporc + tests
- Packaging de l'app : windows et fedora, pytorch GPU et CPU avec pyinstaller et python embed + comparaison
- Découverte de onnx : plus léger pour l'inférence -> inférence avec et packaging à nouveau + comparaison
- Choix selon comparaison : onnx et python embed sur windows et pyinstaller sur fedora + porjet script installation env de dev
- Grosse phase de developpement : 
    - Gérer model t1 /flair
    - BIDS
    - disclaimer + about + help 
    - Choix espace MNI ou patient
    - Stop réactif
    - charger un modele, tester un modele local (cli uniquement), maj liste modèle dans .ini
    - gestion des viewers
    - threshold + pmap (options)
    - sauvegarde brain extraction
    - brain extraction only
- creation d'un script pour créer env de dev sous linux
- Docstring + commenter + documentation utlisateure
- création d'un exe pour lancer le programme python avec python embed
J'en suis ici 

## Evaluation du travail/Bilan/Retour d'expérience
Retour sur la méthode de travail ? les choix des technos en fonction du temps de stage ? L'app finale : taille performance utilisation ? Comment j'aurai fait si je reprend à 0

## Conclusion



# Outline of the report

1 - INTRODUCTION

2 - WORK ENVIRONMENT AND INTERNSHIP TOPIC

- lab
- team
- supervisors
- CSR analysis (asked)
- internship topic presentation

3 - CONTEXT ANALYSIS AND WORK PLAN 

- main objective
- medical and technical context : 
    - existing model, nnUNet
    - preprocessing and postprocessing performed
- requested features (The list of feature)
- possible technological solutions and choices :
    - most likely Python but try C++
    - full pipeline: preprocess, inference, post-processing
    - packaging: pip, python embed, pyinstaller, etc.
    - command line app, but a graphical app is also possible
- work plan : 
    - literature and technology review
    - requirements review
    - experiments and prototyping
    - architecture design
    - in parallel :
        - application development
        - packaging and deployment
        - testing and comparison
        - documentation
    - final delivery and internship report

4 - WORK DONE

- literature and technology review
- requirements review
- experiments and prototyping
- architecture design
- application development
- packaging and deployment
- testing and comparison
- documentation
- final delivery

5 - EVALUATION AND FEEDBACK

- Work method
- Technology choices
- Application result
- What I would change

6 - CONCLUSION
