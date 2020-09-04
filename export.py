from train import get_model
import torch
import os
import argparse

def build_argparser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", help="number of classes", required=True, type=int)
    parser.add_argument("--size", help="size of input image", required=True, type=int)
    parser.add_argument("--model_weights", help="path to the model weights", required=True, type=str)
    parser.add_argument("--output_path", help="path to the onnx model file", required=True, type=str)
    parser.add_argument("--input_names", nargs='+', help="list of input names", default=["image"])
    parser.add_argument("--output_names", nargs='+', help="list of output names", default=["boxes", "labels", "scores"])
    
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    
    #num_classes = 2 # (bread/not bread)
    #size = 512
    #model_weights = "./model/bread_detector_epoch000.pt"
    #output_path = "./onnx/bread_detector.onnx"
    #input_names = ["image"]
    #output_names = ["boxes", "labels", "scores"]
    num_classes = args.num_classes # (bread/not bread)
    size = args.size
    model_weights = args.model_weights
    output_path = args.output_path
    input_names = args.input_names
    output_names = args.output_names
    
    input_shape = [3, size, size]
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    
    dummy_input = [torch.rand(input_shape)]
    
    torch.onnx.export(model, dummy_input, output_path, input_names=input_names, output_names=output_names, opset_version=11)