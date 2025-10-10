# For these files only sample file is provided at "/data-files/esmmd/" is provided, update the path according to the original file.
DICT_TRAIN_PATH = "/path-to-file/dict_train_esmmd.json"
DICT_TEST_PATH = "/path-to-file/dict_test_esmmd.json"
DISEASE_SYMPTOM_PATH = "/path-to-file/disease_symptom_esmmd.json"

#For these files the entire file is available at /data-files/esmmd/
SYMPTOM_COOCCURRENCE_PATH = "/data-files/esmmd/symptom_co-occurence_esmmd.json" 
P_D_GIVEN_S_PATH = f"/data-files/esmmd/Reqd_wts_esmmd_2.csv"

# These are the image files download the files and keep all the train images in a single folder and test in another single folder
TRAIN_IMAGE_DIR = f"/image-data-path/data-files/all_images/train/"
TEST_IMAGE_DIR = f"/image-data/all_images/test/"

# Download this file from https://drive.google.com/file/d/1lwfErF8Q0O7Gpyo_U7feCicYNG0m0KLc/view?usp=sharing and provide the path here
IMAGE_EMBEDDINGS_PATH = f"/embedding-path/vgg19_image_embeddings.npz"

#Update with the output paths
RESULTS_OUTPUT_PATH = f"/output-path/result_esmmd_mod1.csv"
GRAPH_EDGE_DIR = "/output-path/graph-edges/"
CLIP_MODEL_NAME = "ViT-B/32"

# PageRank parameters
PPR_C = 0.15
PPR_EPSILON = 1e-9
PPR_BETA = 0.6
PPR_GAMMA = 0.4
PPR_MAX_ITERS = 100
RANK_THRESHOLD = 0.9

# Dialogue parameters
MAX_DIALOGUE_TURNS = 8
TOP_K_COOCCURRENCE = 10
TOP_K_ENTROPY = 5

# Image-to-symptom parameters
IMAGE_SIMILARITY_WEIGHT = 0.25
IMAGE_EXACT_MATCH_WEIGHT = 0.75
IMAGE_DISEASE_THRESHOLD = 0.1

DONT_KNOW_THRESHOLD = 0.92
# Disease pruning threshold
MIN_EDGE_WEIGHT_THRESHOLD = 0.02

VISUAL_SYMPTOMS = [
    'eye redness', 'eyelid lesion or rash', 'foot or toe swelling',
    'hand or finger lump or mass', 'itchy eyelid', 'knee swelling',
    'lip swelling', 'mouth ulcer', 'neck swelling', 'redness in ear',
    'skin dryness , peeling , scaliness', 'mass on eyelid',
    'skin growth', 'skin rash', 'swollen eye', 'swollen or red tonsils',
    'rash', 'skin lesion', 'abnormal appearing skin', 'warts',
    'diaper rash', 'acne or pimples', 'wrist swelling', 'swollen tongue',
    'hand or finger swelling', 'itching of skin', 'allergic reaction',
    'skin swelling', 'skin irritation', 'skin moles',
    'wrist lump or mass', 'knee lump or mass', 'white discharge from eye',
    'eyelid swelling', 'arm lump or mass'
]
