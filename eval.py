import torch
from train import get_model, get_transform
from data.dataset import BreadDataset
from PIL import Image, ImageDraw
import numpy as np
import os
import argparse

def build_argparser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", help="number of classes", required=True, type=int)
    parser.add_argument("--coco_annotation", help="path to the coco-style json file", required=True, type=str)
    parser.add_argument("--size", help="size of input image", required=True, type=int)
    parser.add_argument("--model_weights", help="path to the model weights", required=True, type=str)
    parser.add_argument("--fig_dir", help="directory for storing prediction results", required=True, type=str)
                
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    
    #num_classes = 2 # (bread/not bread)
    #size = 512
    #coco_annotation = "./trainval.json"
    #model_weights = "./model/bread_detector_epoch002.pt"
    #fig_dir = "./fig/model_1/eval"
    num_classes = args.num_classes # (bread/not bread)
    size = args.size
    coco_annotation = args.coco_annotation
    model_weights = args.model_weights
    fig_dir = args.fig_dir
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    
    pic_count = 0
    dataset = BreadDataset(coco_annotation, size, transforms=get_transform(False))
    for image, targets in dataset:
        bboxes = targets["boxes"]
        
        with torch.no_grad():
            prediction = model([image])
            
        draw_image = Image.fromarray(image.mul(255).permute(1, 2,0).byte().numpy())
        draw = ImageDraw.Draw(draw_image)
        
        for elem in range(len(bboxes)):
            draw.rectangle([(bboxes[elem][0], bboxes[elem][1]), (bboxes[elem][2], bboxes[elem][3])], outline ="green", width =3)

        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals= 4)
           
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))
        
        filename = "{:04d}.jpg".format(pic_count)
        draw_image.save(os.path.join(fig_dir, filename), "JPEG")
        pic_count += 1
        #break 