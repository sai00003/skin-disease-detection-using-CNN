Download the dataset from here https://www.kaggle.com/datasets/shubhamgoel27/dermnet
place the dataset like this (place both the project folder and datset in a same directory)
  anyfolder
    |
    |-project
    |    |-static
    |    |-...
    |-dataset
    |
In main.py 
  class_labels = [
    'Acne', 'Atopic Dermatitis Photos', 'Benign Tumors', 'Bullous Disease',
    'Eczema', 'Exanthems', 'Herpes HPV', 'Melanoma Skin Cancer',
    'Nail Fungus', 'Scabies Lyme Disease', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos'
]
place disease names according to the order of your dataset subfolders 
Hope this will help you in doing your college assignment.
