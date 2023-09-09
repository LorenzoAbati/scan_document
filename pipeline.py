import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
import re
import csv
from tqdm import tqdm

model_name = input('model name: ')


# Dataset creation
class RectangleDataset(Dataset):
    def __init__(self, img_dir, hashmap, transform=None):
        super(RectangleDataset, self).__init__()
        self.img_dir = img_dir
        self.image_names = list(hashmap.keys())
        self.coords = list(hashmap.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.image_names[idx]}.jpg"
        image = io.read_image(img_name).float() / 255.0
        if self.transform:
            image = self.transform(image)
        coords = torch.Tensor(self.coords[idx])
        return image, coords


# Updated Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Updated CNN model
class RectangleNet(nn.Module):
    def __init__(self):
        super(RectangleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.maxpool(self.bn1(self.relu(self.conv1(x))))
        x = self.maxpool(self.bn2(self.relu(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Reshapes the tensor for the fully connected layer
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, coords in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"batch loss": loss.item()})
        val_loss = evaluate(model, dataloader_test, criterion)
        scheduler.step(val_loss) # Adjust learning rate based on validation loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(dataloader)}, Val Loss: {val_loss}")


def extract_info_from_line(line):
    description = line[0]
    img_path = line[2]

    # Extract x, y, width, and height values from the description
    match = re.search(r'in position \[ (\d+) , (\d+) \]', description)
    if match:
        x, y = map(int, match.groups())
    else:
        return None

    match = re.search(r'of size (\d+) x (\d+)', description)
    if match:
        x2, y2 = map(int, match.groups())
    else:
        return None

    # Extract image number from img_path
    img_num = re.search(r'(\d+).jpg', img_path)
    if img_num:
        img_num = img_num.group(1)
    else:
        return None

    return (img_num, (x, y, x2, y2))


# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, coords in dataloader:
            outputs = model(images)
            loss = criterion(outputs, coords)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# train dataset

hashmap = {}

# Replace with the path to your csv file
file_path = 'data/rectangle_ellipse_multimodal/train.csv'
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter='|')
    for line in reader:
        info = extract_info_from_line(line)
        if info:
            hashmap[info[0]] = info[1]


# test dataset

hashmap_test = {}

file_path_test = 'data/rectangle_ellipse_multimodal/test.csv'
with open(file_path_test, 'r') as file:
    reader = csv.reader(file, delimiter='|')
    for line in reader:
        info = extract_info_from_line(line)
        if info:
            hashmap_test[info[0]] = info[1]

dataset_test = RectangleDataset("data/rectangle_ellipse_multimodal/test/rectangle", hashmap_test, transform=test_transforms)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)


# Training pipeline
dataset = RectangleDataset("data/rectangle_ellipse_multimodal/train/rectangle", hashmap, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = RectangleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

train(model, dataloader, criterion, optimizer, scheduler, num_epochs=10)


# Evaluate on test set
test_loss = evaluate(model, dataloader_test, criterion)
print(f"Test Loss: {test_loss}")

# Save the model after training
model_save_path = f"{model_name}.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
