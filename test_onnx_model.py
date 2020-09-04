import onnxruntime as rt
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import glob
import os
from tqdm import tqdm

if __name__ == "__main__":
    
    onnx_model_path = "./onnx/bread_detector.onnx"
    image_folder = r"D:\KelvinWu\Datasets\麵包\test dataset"
    fig_dir = "./fig/epoch000/onnx"
    size = 512 # input shape (3, 512, 512)
    threshold = 0.99
    
    input_names = ["image"]
    output_names = ["boxes", "labels", "scores"]
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    sess = rt.InferenceSession(onnx_model_path)
    print(sess.get_outputs()[1].name)
    
    pic_count = 0
    image_files = glob.glob(os.path.join(image_folder, "*.JPG"))
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
        
        filename = "{:04d}.jpg".format(pic_count)
        draw_image.save(os.path.join(fig_dir, filename), "JPEG")
        pic_count += 1
        
        #pred = sess.run([output_name], {input_name: input})[0]