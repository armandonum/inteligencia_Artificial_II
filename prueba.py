import os
import torch
from torch.utils.data import DataLoader, Subset
from mlp import PlantDataset, MLP, load_checkpoint, transform
from tqdm import tqdm

# Verificar si se dispone de una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas a las carpetas descomprimidas
root_dirs = [
    './dataset/descomprimido/achira',
    './dataset/descomprimido/calanchoe',
    './dataset/descomprimido/diente_de_leon',
    './dataset/descomprimido/flor_de_rosa',
    './dataset/descomprimido/palta'
]

# Crear dataset y dataloader
dataset = PlantDataset(root_dirs, transform=transform)

# Configurar el porcentaje del dataset para pruebas (por ejemplo, 80%)
nuevo_porcentaje = 0.8
subset_indices_nuevo = torch.randperm(len(dataset))[:int(nuevo_porcentaje * len(dataset))]
nuevo_subset_dataset = Subset(dataset, subset_indices_nuevo)
nuevo_dataloader = DataLoader(nuevo_subset_dataset, batch_size=64, shuffle=True, num_workers=4, prefetch_factor=2)

# Cargar el modelo y el optimizador desde el checkpoint
model = MLP().to(device)
checkpoint_path = './Checkpoints/mlp_checkpoint_latest_500.pth'
start_epoch = load_checkpoint(checkpoint_path)

# Evaluar el modelo en el nuevo subconjunto de datos
model.eval()
corrects = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(nuevo_dataloader, desc="Evaluación", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        corrects += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = corrects / total
print(f"Precisión en el nuevo subconjunto de datos: {accuracy:.4f}")
