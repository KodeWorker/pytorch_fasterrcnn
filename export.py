from train import get_model
import torch
import os

if __name__ == "__main__":
    
    num_classes = 2 # (bread/not bread)
    model_weights_path = "./model/bread_detector_epoch000.pt"
    output_dir = "./onnx"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    dummy_input = [torch.rand(3, 512, 512)]
    
    input_names = ["image"]
    output_names = ["boxes", "labels", "scores"]
    
    torch.onnx.export(model, dummy_input, os.path.join(output_dir, "bread_detector.onnx"), input_names=input_names, output_names=output_names, opset_version=11)