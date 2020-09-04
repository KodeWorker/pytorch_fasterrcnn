import torch
import json
from PIL import Image
import numpy as np
from PIL import ImageOps

class BreadDataset(torch.utils.data.Dataset):
    
    def __init__(self, coco_annotation, size, root=None, transforms=None):
        super(BreadDataset, self).__init__()
        self.coco_annotation = coco_annotation
        self.size = size
        self.root = root
        self.transforms = transforms
        
        self.parseCOCO_1pic1obj() # temp
    
    def parseCOCO_1pic1obj(self):
        with open(self.coco_annotation, "r", encoding="utf-8") as readFile:
            coco = json.load(readFile)
        
        # parse images
        if self.root:
            self.images = [os.path.join(self.root, img["file_name"]) for img in coco["images"]]
        else:
            self.images = [img["file_name"] for img in coco["images"]]
        # parse boxes 
        self.boxes = [[[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ann["bbox"][2], ann["bbox"][1]+ann["bbox"][3]]] for ann in coco["annotations"]]
        #self.boxes = [[[ann["bbox"][1], ann["bbox"][0], ann["bbox"][1]+ann["bbox"][3], ann["bbox"][0]+ann["bbox"][2]]] for ann in coco["annotations"]]
        #self.boxes = [ann["bbox"] for ann in coco["annotations"]]
        
        # parse categories
        self.categories = [[ann["category_id"]+1] for ann in coco["annotations"]]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = ImageOps.exif_transpose(img)
        
        h, w = img.size
        ratio = self.size / max(w, h)
        
        if w > h:
            img = img.resize((int(h*ratio), self.size), Image.ANTIALIAS)
        elif w < h:
            img = img.resize((self.size, int(w*ratio)), Image.ANTIALIAS)
        else:
            img = img.resize((self.size,self.size), Image.ANTIALIAS)
        
        num_objs = len(self.boxes[idx])
        boxes = torch.as_tensor(np.array(self.boxes[idx])*ratio, dtype=torch.float32)
        #print(boxes)
        labels = torch.as_tensor(self.categories[idx], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) * ratio *ratio
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
      
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target