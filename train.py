# reference: https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.dataset import BreadDataset
import transforms as T
import utils
from engine import train_one_epoch, evaluate

def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
   
if __name__ == "__main__":
    num_classes = 2 # (bread/not bread)
    coco_annotation = "./trainval.json"
    size = 512
    val_ratio = 0.2
    num_workers = 4
    batch_size = 6
    shuffle = True
    num_epochs = 10
    model_dir = "./model"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    dataset = BreadDataset(coco_annotation, size)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - int(len(dataset)*val_ratio), int(len(dataset)*val_ratio)])
    train_set.dataset.transforms = get_transform(train=True)
    val_set.dataset.transforms = get_transform(train=False)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=utils.collate_fn)
    
    print("We have: {} examples, {} are training and {} validation".format(len(dataset), len(train_set), len(val_set)))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_loader, device=device)
        torch.save(model.state_dict(), os.path.join(model_dir, "bread_detector_epoch{:03d}.pt".format(epoch)))
        
    torch.save(model.state_dict(), os.path.join(model_dir, "bread_detector_last.pt"))