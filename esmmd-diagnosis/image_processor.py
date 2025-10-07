"""
Image Processor Module
Handles image-based symptom detection using CLIP and VGG19 embeddings
"""

import torch
import clip
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import *


class ImageProcessor:
    """Processes medical images to extract visual symptoms and diseases"""
    
    def __init__(self, vgg19_embeddings_path, p_d_given_s):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {self.device}...")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
        
        # Tokenize visual symptoms
        self.symptom_list = VISUAL_SYMPTOMS
        self.text_inputs = clip.tokenize(self.symptom_list).to(self.device)
        print(f"✓ CLIP model loaded with {len(self.symptom_list)} visual symptoms")
        
        # Load VGG19 embeddings
        try:
            self.vgg19_embeddings = np.load(vgg19_embeddings_path)
            print(f"✓ VGG19 embeddings loaded")
        except FileNotFoundError:
            print(f"⚠ Warning: VGG19 embeddings not found")
            self.vgg19_embeddings = None
        
        self.p_d_given_s = p_d_given_s
        
        # self.disease_img = {
        #     'Dyshidrosis': ['S11316', 'S31110', 'S3114', 'S31221', 'S31117'],
        #     'Conductive hearing loss': ['S2231', 'S18136', 'S2236', 'S2221'],
        #     'Conjunctivitis due to allergy': ['S1213', 'S5126', 'S513', 'S5110', 'S1217', 'S5111', 'S5113'],
        #     'Drug reaction': ['S31214', 'S51312', 'S3111', 'S3115', 'S3116', 'S3117', 'S3118', 'S5132', 'S31112'],
        #     'Corneal disorder': ['S111', 'S5124', 'S5328', 'S522', 'S524', 'S5223', 'S7436'],
        #     'Eczema': ['S31125', 'S11119', 'S31121', 'S1125', 'S3112', 'S1132', 'S1133', 'S31138', 'S31137', 'S11329', 'S31114', 'S31127'],
        #     'Actinic keratosis': ['S11126', 'S11113', 'S11112', 'S1124'],
        #     'Acne': ['S3122', 'S31215', 'S31214', 'S1137', 'S31223', 'S439'],
            
        # }
        self.disease_img = {'Dyshidrosis': ['$S11_3_16', '$S31_1_10', '$S31_1_4', '$S31_2_21', '$S31_1_17'], 'Conductive hearing loss': ['$S22_3_1', '$S181_3_6', '$S22_3_6', '$S22_2_1'], 'Conjunctivitis due to allergy': ['$S1_2_13', '$S5_1_26', '$S5_1_3', '$S5_1_10', '$S1_2_17', '$S5_1_11', '$S5_1_13'], 'Drug reaction': ['$S31_2_14', '$S51_3_12', '$S31_1_1', '$S31_1_5', '$S31_1_6', '$S31_1_7', '$S31_1_8', '$S51_3_2', '$S31_1_12'], 'Corneal disorder': ['$S1_1_1', '$S5_1_24', '$S5_3_28', '$S5_2_2', '$S5_2_4', '$S5_2_23', '$S74_3_6'], 'Eczema': ['$S31_1_25', '$S11_1_19', '$S31_1_21', '$S11_2_5', '$S31_1_2', '$S11_3_2', '$S11_3_3', '$S31_1_38', '$S31_1_37', '$S11_3_29', '$S31_1_14', '$S31_1_27'], 'Actinic keratosis': ['$S11_1_26', '$S11_1_13', '$S11_1_12', '$S11_2_4'], 'Acariasis': ['$S31_1_22', '$S31_3_3', '$S31_1_8', '$S31_1_12', '$S31_1_27', '$S31_1_4', '$S31_1_7', '$S31_1_2'], 'Chickenpox': ['$S31_1_36', '$S31_1_8', '$S31_1_12', '$S5_3_2', '$S31_1_7', '$S31_1_10', '$S31_1_16', '$S5_3_10'], 'Gout': ['$S50_1_31', '$S205_3_1', '$S205_3_3', '$S50_2_3', '$S50_2_4'], 'Contact dermatitis': ['$S31_1_19', '$S31_1_9', '$S31_1_10', '$S1_3_12', '$S31_1_37', '$S11_3_5', '$S1_3_15', '$S31_1_14'], 'Cat scratch disease': ['$S31_2_13', '$S31_2_1', '$S159_1_1', '$S159_1_9', '$S31_2_21', '$S159_1_14'], 'Factitious disorder': ['$S51_1_19', '$S51_2_1', '$S37_1_1', '$S51_2_6', '$S51_1_26', '$S51_2_5'], 'Ectropion': ['$S1_3_21', '$S74_3_1', '$S74_1_1', '$S1_3_1', '$S1_3_3', '$S5_2_2', '$S5_2_3', '$S1_3_6', '$S1_3_13', '$S5_2_20', '$S5_2_22'], 'Endophthalmitis': ['$S5_1_21', '$S1_2_31', '$S5_1_1', '$S1_2_2', '$S1_2_1', '$S5_1_2', '$S1_3_4', '$S5_1_4', '$S1_2_6', '$S1_2_35', '$S5_1_8', '$S5_1_9', '$S1_2_15', '$S1_2_19', '$S1_2_16'], 'Corneal abrasion': ['$S1_3_10', '$S5_2_1', '$S1_3_11', '$S1_3_13', '$S5_2_12', '$S1_2_12', '$S5_2_14', '$S5_2_16', '$S1_3_14', '$S5_2_19', '$S5_2_21'], 'Allergy': ['$S1_1_1', '$S1_3_23', '$S31_1_3', '$S31_1_6', '$S31_1_13'], 'Erythema multiforme': ['$S108_3_1', '$S37_1_1', '$S31_1_2', '$S108_3_4', '$S108_3_5', '$S108_3_6', '$S31_1_11', '$S31_1_15', '$S31_1_17'], 'Cyst of the eyelid': ['$S74_2_1', '$S74_2_5', '$S4_2_6', '$S74_2_6', '$S31_3_6'], 'Ganglion cyst': ['$S4_2_1', '$S4_2_2', '$S39_3_1', '$S39_3_3', '$S205_3_7'], 'Diabetic retinopathy': ['$S5_3_3', '$S5_3_4', '$S5_3_9'], 'Diaper rash': ['$S31_3_4'], 'Graves disease': ['$S50_3_1'], 'Chalazion': ['$S5_2_30', '$S74_2_1', '$S5_2_18', '$S74_3_5', '$S74_2_7', '$S4_3_8'], 'Dermatitis due to sun exposure': ['$S4_3_24'], 'Chondromalacia of the patella': ['$S205_3_2'], 'Acne': ['$S31_2_2', '$S31_2_15', '$S31_2_14', '$S11_3_7', '$S31_2_23', '$S4_3_9'], 'Flat feet': ['$S4_3_5'], 'Aphakia': ['$S5_3_8'], 'Acanthosis nigricans': ['$S4_3_7'], 'Diabetes insipidus': ['$S37_3_1']}

    def _clean_image_name(self, img_name):
        cleaned = img_name.replace('$', '')
        if not cleaned.lower().endswith('.jpg'):
            cleaned += '.jpg'
        return cleaned

    def img_to_sym(self, image_filename):
        """Extract visual symptoms using CLIP"""
        train_path = os.path.join(TRAIN_IMAGE_DIR, image_filename)
        test_path = os.path.join(TEST_IMAGE_DIR, image_filename)
        
        if os.path.exists(train_path):
            image_path = train_path
        elif os.path.exists(test_path):
            image_path = test_path
        else:
            raise FileNotFoundError(f"Image not found: {image_filename}")
        
        # Determine top-k
        if "_1" in image_filename:
            top_k = 5
        elif "_2" in image_filename:
            top_k = 10
        else:
            top_k = 15
        
        # Process with CLIP
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            top_probs, top_labels = similarity[0].topk(min(top_k, len(self.symptom_list)))
            top_indices = top_labels.cpu().tolist()
            
            top_symptoms = [self.symptom_list[i] for i in top_indices if i < len(self.symptom_list)]
            return top_symptoms
    
    def image_to_dis(self, img_name, similarity_weight=IMAGE_SIMILARITY_WEIGHT, 
                    exact_match_weight=IMAGE_EXACT_MATCH_WEIGHT):
        """
        Map image to diseases using VGG19 embeddings and disease-image mapping
        """
        # Clean image name for lookups
        cleaned_img_name = self._clean_image_name(img_name)
        
        print(f"    [DEBUG] Input: {img_name}")
        print(f"    [DEBUG] Cleaned: {cleaned_img_name}")
        
        # Fallback if VGG19 not available
        if self.vgg19_embeddings is None:
            print(f"    [DEBUG] VGG19 embeddings not loaded!")
            symptoms = self.img_to_sym(img_name)
            diseases = self.img_sym_dis(symptoms)
            return {d: 0.5 for d in diseases}
        
        # Check if image exists in VGG19 embeddings
        if cleaned_img_name not in self.vgg19_embeddings.files:
            # print(f"    [DEBUG] '{cleaned_img_name}' NOT found in VGG19")
            # print(f"    [DEBUG] Sample VGG19 files: {list(self.vgg19_embeddings.files)[:5]}")
            # FALLBACK: Use CLIP symptoms instead
            symptoms = self.img_to_sym(img_name)
            diseases = self.img_sym_dis(symptoms)
            # print(f"    [DEBUG] Fallback to CLIP: {len(diseases)} diseases found")
            return {d: 0.5 for d in diseases} if diseases else {}
        
        # print(f"    [DEBUG] '{cleaned_img_name}' found in VGG19!")
        
        # Rest of your code...
        input_embedding = self.vgg19_embeddings[cleaned_img_name]
        disease_scores = {}
        
        for disease, images in self.disease_img.items():
            cleaned_images = [self._clean_image_name(img) for img in images]
            disease_embeddings = []
            for img in cleaned_images:
                if img in self.vgg19_embeddings.files:
                    disease_embeddings.append(self.vgg19_embeddings[img])
            
            if not disease_embeddings:
                continue
            
            sim_scores = cosine_similarity([input_embedding], disease_embeddings)[0]
            avg_similarity = np.mean(sim_scores)
            exact_match_score = 0.8 if cleaned_img_name in cleaned_images else 0.0
            final_score = (similarity_weight * avg_similarity + 
                        exact_match_weight * exact_match_score)
            disease_scores[disease] = final_score
        
        if not disease_scores:
            print(f"    [DEBUG] No disease scores computed!")
            return {}
        
        # Normalize
        scores = np.array(list(disease_scores.values()))
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-8:
            normalized_scores = {disease: 0.5 for disease in disease_scores}
        else:
            normalized_scores = {
                disease: (score - min_score) / (max_score - min_score)
                for disease, score in disease_scores.items()
            }
        
        sorted_diseases = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"    [DEBUG] Top 5 diseases: {[d for d, s in sorted_diseases[:5]]}")
        return dict(sorted_diseases[:5])

