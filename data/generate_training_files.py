import json
import glob
from tqdm import tqdm
import os

if __name__ == "__main__":
    
    label_dir = r"D:\KelvinWu\Datasets\麵包\labelme_annotations"
    second_dir = "D:\KelvinWu"
    label_files = glob.glob(os.path.join(label_dir, "*.json"))
    output_dir = "../images"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    record = {}
    
    for label_file in tqdm(label_files):
        label = json.load(open(label_file, "r", encoding="utf-8"))
        
        if os.path.exists(os.path.abspath(os.path.join(label_dir, label["imagePath"]))):
            image_path = os.path.abspath(os.path.join(label_dir, label["imagePath"]))
        else:   
            image_path = os.path.abspath(os.path.join(second_dir, label["imagePath"]))
            
        label["imagePath"] = image_path
        
        with open(os.path.join(output_dir, os.path.basename(label_file)), "w") as writeFile:
            json.dump(label, writeFile)