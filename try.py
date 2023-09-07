import torch
import torch.nn as nn
from torchvision import transforms, io
import matplotlib.pyplot as plt
import numpy as np


class RectangleNet(nn.Module):
    def __init__(self):
        super(RectangleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def draw_rectangle(image, coordinates):
    plt.imshow(image)
    plt.gca().add_patch(
        plt.Rectangle((coordinates[0], coordinates[1]),
                      coordinates[2] - coordinates[0],
                      coordinates[3] - coordinates[1],
                      linewidth=1, edgecolor='lime', facecolor='none'))
    plt.show()


def predict_rectangle_coordinates(img_path, model_path):
    # Load image and preprocess
    image = io.read_image(img_path).float() / 255.0
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Create a model instance and load the trained weights
    model = RectangleNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict rectangle coordinates
    with torch.no_grad():
        coords = model(image)
        coords = coords.squeeze().tolist()

    return coords


if __name__ == "__main__":
    img_path = input("Enter the path of the image: ")
    model_path = "new_test3ep.pth"

    coordinates = predict_rectangle_coordinates(img_path, model_path)
    print(f"Predicted Rectangle Coordinates: {coordinates}")

    # Convert the image to NumPy array and normalize it
    image_np = np.array(io.read_image(img_path).float() / 255.0).transpose(1, 2, 0)

    # Draw the rectangle on the image
    draw_rectangle(image_np, coordinates)
