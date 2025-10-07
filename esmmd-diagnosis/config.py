# JSON data files
DICT_TRAIN_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/dict_train_esmmd.json"
DICT_TEST_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/dict_test_esmmd.json"
SYMPTOM_COOCCURRENCE_PATH = "/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/symptom_co-occurence_esmmd.json"
DISEASE_SYMPTOM_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/disease_symptom_esmmd.json"

# CSV files
P_D_GIVEN_S_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/Reqd_wts_esmmd_2.csv"

# Image paths
TRAIN_IMAGE_DIR = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/all_images/train/"
TEST_IMAGE_DIR = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/all_images/test/"

IMAGE_EMBEDDINGS_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/data-files/ESMMD-updated-files/vgg19_image_embeddings.npz"

# Output paths
RESULTS_OUTPUT_PATH = f"/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/result_esmmd_mod1.csv"
GRAPH_EDGE_DIR = "/home/Byomakesh/ours-diagnosis/esmmd-diagnosis/graph-edges/"

# ============================================
# MODEL PARAMETERS
# ============================================

# CLIP model
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
