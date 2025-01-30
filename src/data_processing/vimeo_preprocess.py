import os
import cv2

def preprocess_vimeo(input_dir="data\\vimeo_triplet", output_dir="data\\processed_frames"):
    # Load training split list
    split_file = os.path.join(input_dir, "tri_trainlist.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"{split_file} not found. Ensure it's in the Vimeo90K root directory.")
    
    with open(split_file, "r") as f:
        triplets = [line.strip() for line in f.readlines()]

    for triplet in triplets:
        seq_dir = os.path.join(input_dir, "sequences", triplet)
        output_seq_dir = os.path.join(output_dir, triplet)
        os.makedirs(output_seq_dir, exist_ok=True)

        # Process all 3 frames (im1.png, im2.png, im3.png)
        for frame_name in ["im1.png", "im2.png", "im3.png"]:
            img_path = os.path.join(seq_dir, frame_name)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} does not exist. Skipping...")
                continue
            
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (256, 256))  # Resize to 256x256
            cv2.imwrite(os.path.join(output_seq_dir, frame_name), img_resized)

if __name__ == "__main__":
    preprocess_vimeo()