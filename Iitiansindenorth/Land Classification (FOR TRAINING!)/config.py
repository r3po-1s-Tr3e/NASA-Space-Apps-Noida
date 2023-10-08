IDX_CLASS_LABELS = {
    0: 'AnnualCrop',
    1: 'Forest', 
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}

#PATH = 'check/image.jpg'
PATH = r"C:\Users\hryad\Desktop\iitm\Space Apps\Land-Cover-Classification-using-Sentinel-2-Dataset\data\2750\Industrial\Industrial_500.jpg"
MODEL_PATH = r'data/model'
NUM_CLASSES = 10
DATA_TESTING_PATH = r"C:\Users\hryad\Desktop\shit\data\test\SeaLake"
TEST_DATASET_PATH = r"C:\Users\hryad\Desktop\iitm\Space Apps\Land-Cover-Classification-using-Sentinel-2-Dataset\data\test" # Assuming you have a 'test' folder with images