import glob
from train import get_model
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import ToTensor
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

def build_argparser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", help="number of classes", required=True, type=int)
    parser.add_argument("--size", help="size of input image", required=True, type=int)
    parser.add_argument("--image_folder", help="directory for testing images", required=True, type=str)
    parser.add_argument("--model_weights", help="path to the model weights", required=True, type=str)
    parser.add_argument("--fig_dir", help="directory for storing prediction results", required=True, type=str)
    parser.add_argument("--threshold", help="class score threshold", required=True, type=float)
    parser.add_argument("--format", help="image format", required=True, type=str)
    
    return parser
    
if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    #num_classes = 2 # (bread/not bread)
    #size = 512
    #image_folder = r"D:\KelvinWu\Datasets\麵包\test dataset"
    #model_weights = "./model/bread_detector_epoch000.pt"
    #fig_dir = "./fig/epoch000/test"
    #threshold = 0.99
    num_classes = args.num_classes # (bread/not bread)
    size = args.size
    image_folder = args.image_folder
    model_weights = args.model_weights
    fig_dir = args.fig_dir
    threshold = args.threshold
    format = args.format
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    
    transform = ToTensor()
    
    image_files = glob.glob(os.path.join(image_folder, "*.{}".format(format)))
    for image_file in tqdm(image_files):
        
        draw_image = Image.open(image_file)
        draw_image = ImageOps.exif_transpose(draw_image)
        
        h, w = draw_image.size
        ratio = size / max(w, h)
        
        if w > h:
            img = draw_image.resize((int(h*ratio), size), Image.ANTIALIAS)
        elif w < h:
            img = draw_image.resize((size, int(w*ratio)), Image.ANTIALIAS)
        else:
            img = draw_image.resize((size,self.size), Image.ANTIALIAS)
        
        image = transform(img)
        
        with torch.no_grad():
            prediction = model([image])
        
        draw = ImageDraw.Draw(draw_image)
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()/ratio
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals= 4)
            #label = prediction[0]["labels"][element].cpu().numpy()
            #print(label)
            
            if score > threshold:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))
        
        filename = os.path.basename(image_file)
        draw_image.save(os.path.join(fig_dir, filename))
        