import onnxruntime as rt
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import glob
import os
from tqdm import tqdm

import argparse

def build_argparser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--size", help="size of input image", required=True, type=int)
    parser.add_argument("--image_folder", help="directory for testing images", required=True, type=str)
    parser.add_argument("--onnx_model", help="path to the onnx model file", required=True, type=str)
    parser.add_argument("--fig_dir", help="directory for storing prediction results", required=True, type=str)
    parser.add_argument("--threshold", help="class score threshold", required=True, type=float)
    parser.add_argument("--input_names", nargs='+', help="list of input names", default=["image"])
    parser.add_argument("--output_names", nargs='+', help="list of output names", default=["boxes", "labels", "scores"])
    parser.add_argument("--format", help="image format", required=True, type=str)
    
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    #onnx_model = "./onnx/bread_detector.onnx"
    #image_folder = r"D:\KelvinWu\Datasets\麵包\test dataset"
    #fig_dir = "./fig/epoch000/onnx"
    #size = 512 # input shape (3, 512, 512)
    #threshold = 0.99    
    #input_names = ["image"]
    #output_names = ["boxes", "labels", "scores"]
    onnx_model = args.onnx_model
    image_folder = args.image_folder
    fig_dir = args.fig_dir
    size = args.size
    threshold = args.threshold
    input_names = args.input_names
    output_names = args.output_names
    format = args.format
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    sess = rt.InferenceSession(onnx_model)
    print(sess.get_outputs()[1].name)
    
    image_files = glob.glob(os.path.join(image_folder, "*.{}".format(format)))
    for image_file in tqdm(image_files):
        
        draw_image = Image.open(image_file)
        draw_image = ImageOps.exif_transpose(draw_image)
        
        h, w = draw_image.size
        ratio = size / max(w, h)
        
        draw_image = ImageOps.pad(draw_image, (max(w, h), max(w, h)), method=3, color=None, centering=(0.5, 0.5))
        img = draw_image.resize((size,size), Image.ANTIALIAS)
        
        image = np.uint8(img).transpose(2,0,1) / 255
        image = image.astype(np.float32)
        
        pred = sess.run(output_names, {input_names[0]: image})
        
        draw = ImageDraw.Draw(draw_image)
        for element in range(len(pred[0])):
            boxes = pred[0][element]/ratio
            score = np.round(pred[2][element], decimals= 4)
            #label = prediction[0]["labels"][element].cpu().numpy()
            #print(label)
            
            if score > threshold:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
                draw.text((boxes[0], boxes[1]), text = str(score))
        
        filename = os.path.basename(image_file)
        draw_image.save(os.path.join(fig_dir, filename))