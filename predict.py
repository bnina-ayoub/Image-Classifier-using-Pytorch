from get_predict_args import get_predict_args
from lab_check_functions import *
import torch
import os

def main():
    args = get_predict_args()
    check_predict_command_line_arguments(args)
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    use_gpu = args.gpu
    if use_gpu :
        device = 'cuda'
    else:
        device = 'cpu'
    data_directory = "flowers"    
    model = load_checkpoint(checkpoint, device)
    processed_image = process_image(os.path.join(data_directory, "test/", image_path))
    processed_image = torch.unsqueeze(torch.from_numpy(processed_image), 0).to(device).float()
    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image.to(device))                         
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
        probs = top_p.tolist()[0]
        class_list = top_class.tolist()[0]
        
        classes = []
    
    classes = class_mapping(category_names, class_list)
    print(probs, classes)

if __name__ == "__main__":
    main()