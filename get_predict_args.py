import argparse

def get_predict_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image with probability")

    # Positional arguments
    parser.add_argument("image_path", type=str, help="Path to the image for prediction")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")

    # Optional arguments
    parser.add_argument("--top_k", type=int, default=3, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the mapping of categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    return parser.parse_args()
