import cv2
import numpy as np
import os

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Folder paths for templates
template_folders = {
    "5 Rupees": "dataset/train/5_rupees",
    "10 Rupees": "dataset/train/10_rupees",
    "20 Rupees": "dataset/train/20_rupees",
    "50 Rupees": "dataset/train/50_rupees",
    "100 Rupees": "dataset/train/100_rupees",
    "200 Rupees": "dataset/train/200_rupees",
    "500 Rupees": "dataset/train/500_rupees",
}

# Load templates
def load_templates():
    template_descriptors = {}
    for label, folder_path in template_folders.items():
        descriptors = []
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)
                kp, desc = orb.detectAndCompute(img, None)
                if desc is not None:
                    descriptors.append(desc)
        template_descriptors[label] = descriptors
    return template_descriptors

template_descriptors = load_templates()

# Currency detection function
def detect_currency(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp_img, desc_img = orb.detectAndCompute(img, None)

    best_match = None
    max_good_matches = 0
    good_match_threshold = 15

    for bill_name, descriptors_list in template_descriptors.items():
        for desc_template in descriptors_list:
            matches = bf.knnMatch(desc_template, desc_img, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match = bill_name

    return best_match if best_match else "Unknown Currency"
