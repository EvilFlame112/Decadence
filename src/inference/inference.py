import cv2
import torch
from torchvision import transforms
import os
import sys
# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import the model
from models.model import InterpolationModel
def interpolate(frame1_path, frame2_path, model_path="models/interpolator.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterpolationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess frames
    transform = transforms.Compose([transforms.ToTensor()])
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    with torch.no_grad():
        input1 = transform(frame1).unsqueeze(0).to(device)
        input2 = transform(frame2).unsqueeze(0).to(device)
        output = model(input1, input2)
        output = output.squeeze().cpu().permute(1, 2, 0).numpy() * 255  # Convert to HWC numpy array

    return output.astype("uint8")

# Example usage
interpolated_frame = interpolate("data/processed_frames/00001/0001/im1.png", "data/processed_frames/00001/0001/im3.png")
cv2.imwrite("outputs/interpolated.png", interpolated_frame)