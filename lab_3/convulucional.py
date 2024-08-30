import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Verificar si se dispone de una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el dataset
class PlantDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_map = {
            'achira': 0,
            'calanchoe': 1,
            'diente_de_leon': 2,
            'flor_de_rosa': 3,
            'palta': 4
        }

        for root_dir in root_dirs:
            class_name = root_dir.split('/')[-1].lower()
            label = self.class_map[class_name]
            for root, _, files in os.walk(root_dir):
                for file_name in files:
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file_name)
                        self.image_paths.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except UnidentifiedImageError as e:
            print(f"Error: No se pudo identificar la imagen en {img_path}. {e}")
            return None, label

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Rutas a las carpetas descomprimidas
root_dirs = [
    '../dataset/descomprimido/achira',
    '../dataset/descomprimido/calanchoe',
    '../dataset/descomprimido/diente_de_leon',
    '../dataset/descomprimido/flor_de_rosa',
    '../dataset/descomprimido/palta'
]

# Crear dataset y dataloader
dataset = PlantDataset(root_dirs, transform=transform)

# Definir un collate_fn personalizado que filtra los valores None
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Definir la red neuronal convolucional (CNN)
class PlantCNN(nn.Module):
    def __init__(self):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Inicializar el modelo
model = PlantCNN().to(device)

# Definir el optimizador y la función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configurar el entrenamiento
epochs = 50
checkpoint_interval = 1
train_losses = []
val_losses = []

# División del dataset en entrenamiento y validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Entrenamiento y validación del modelo
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validación del modelo
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    accuracy = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1]
        }
        torch.save(checkpoint, f"./Checkpoints/cnn_checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint guardado en ./Checkpoints/cnn_checkpoint_epoch_{epoch+1}.pth")

# Graficar la pérdida de entrenamiento y validación
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Pérdida de Entrenamiento')
plt.plot(val_losses, label='Pérdida de Validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el Entrenamiento y Validación')
plt.legend()
plt.show()
