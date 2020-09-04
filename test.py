import glob
from train import get_model
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import ToTensor
import os
import torch
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    num_classes = 2 # (bread/not bread)
    size = 512
    image_folder = r"D:\KelvinWu\Datasets\麵包\test dataset"
    model_weights_path = "./model/bread_detector_epoch000.pt"
    fig_dir = "./fig/epoch000/test"
    threshold = 0.99
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    transform = ToTensor()
    
    pic_count = 0
    image_files = glob.glob(os.path.join(image_folder, "*.JPG"))
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
        
        filename = "{:04d}.jpg".format(pic_count)
        draw_image.save(os.path.join(fig_dir, filename), "JPEG")
        pic_count += 1
        