from data.dataset import BreadDataset
from PIL import ImageDraw

if __name__ == "__main__":
    coco_annotation = "./trainval.json"
    size = 512
    
    dataset = BreadDataset(coco_annotation, size)
    
    x, y = dataset[500]
    
    bbox = y["boxes"].detach().cpu().numpy()[0]
    print(bbox)
    draw = ImageDraw.Draw(x)
    draw.rectangle(bbox, outline="black", width=5)
    
    x.save("demo.jpg", "JPEG")