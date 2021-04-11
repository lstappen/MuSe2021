import os

# adjust your paths here
BASE_PATH = '/Muse2021/data/'

PATH_TO_ALIGNED_FEATURES = {
    'wilder': os.path.join(BASE_PATH, 'c1_muse_wilder/feature_segments'),
    'sent': os.path.join(BASE_PATH, 'c2_muse_sent/feature_segments'),
    'stress': os.path.join(BASE_PATH, 'c3_muse_stress/feature_segments'),
    'physio': os.path.join(BASE_PATH, 'c4_muse_physio/feature_segments')
}

PATH_TO_LABELS = {
    'wilder': os.path.join(BASE_PATH, 'c1_muse_wilder/label_segments'),
    'sent': os.path.join(BASE_PATH, 'c2_muse_sent/label_segments'),
    'stress': os.path.join(BASE_PATH, 'c3_muse_stress/label_segments'),
    'physio': os.path.join(BASE_PATH, 'c4_muse_physio/label_segments')
}

PATH_TO_METADATA = {
    'wilder': os.path.join(BASE_PATH, 'c1_muse_wilder/metadata'),
    'sent': os.path.join(BASE_PATH, 'c2_muse_sent/metadata'),
    'stress': os.path.join(BASE_PATH, 'c3_muse_stress/metadata'),
    'physio': os.path.join(BASE_PATH, 'c4_muse_physio/metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

OUTPUT_PATH = '/Muse2021/results/'
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')
