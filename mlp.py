import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.onnx
import torch.jit

# Verificar si se dispone de una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

class PlantDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Mapeo de nombres de carpetas a etiquetas numéricas

        self.class_map = {
            'achira': 0,
            'calanchoe': 1,
            'diente_de_leon': 2,
            'flor_de_rosa': 3,
            'palta': 4
        }
        # Cargar todas las rutas de imágenes y sus etiquetas
        for root_dir in root_dirs:
            class_name = root_dir.split('/')[-1].lower()  # Asumimos que el nombre de la carpeta es la clase
           # class_name = root_dir.split('/')[-1].lower().replace(' ', '_')
            label = self.class_map[class_name]  # Obtener el label correspondiente
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
            return None, label  # Puedes manejar esto en tu DataLoader para omitir imágenes no válidas


# Rutas a las carpetas descomprimidas
root_dirs = [
    './dataset/descomprimido/achira',
    './dataset/descomprimido/calanchoe',
    './dataset/descomprimido/diente_de_leon',
    './dataset/descomprimido/flor_de_rosa',
    './dataset/descomprimido/palta'
]

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




from torch.utils.data import DataLoader

# Definir un collate_fn personalizado que filtra los valores None
def collate_fn(batch):
    # Filtrar los ejemplos que contienen None
    batch = list(filter(lambda x: x[0] is not None, batch))

    # Si el batch queda vacío después de filtrar, devuelve un tensor vacío
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)

    # Usar el collate por defecto para el resto
    return torch.utils.data.dataloader.default_collate(batch)

# Crear dataset y dataloader
dataset = PlantDataset(root_dirs, transform=transform)
# Crear el DataLoader con el collate_fn personalizado
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, prefetch_factor=2, collate_fn=collate_fn)



import torch
from torch.utils.data import Subset

# Asumiendo que 'dataset' es tu dataset original
subset_indices = torch.randperm(len(dataset))[:int(0.6* len(dataset))]
subset_dataset = Subset(dataset, subset_indices)

# Crea un nuevo DataLoader usando el subset
subset_dataloader = DataLoader(subset_dataset, batch_size=64, shuffle=True, num_workers=4, prefetch_factor=2)



# Prueba con un DataLoader más simple
dataloader_simple = DataLoader(subset_dataset, batch_size=64, shuffle=True)
subset_size_simple = len(subset_dataset)
print(f'Cantidad de datos en el subconjunto con DataLoader simple: {subset_size_simple}')

total_images = len(dataset)
print(f'Cantidad total de imágenes en el dataset: {total_images}')


import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(128 * 128 * 3, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = x.view(-1, 128 * 128 * 3)  # Aplanar las imágenes
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP()




import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Definir el optimizador y la función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configurar el entrenamiento
epochs = 1000
checkpoint_interval = 2
train_losses = []

# Función para calcular la precisión
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = (preds == labels).sum().item()
    accuracy = corrects / len(labels)
    return accuracy

# Cargar el estado del modelo y el optimizador desde el checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch

# Verificar si existe un checkpoint y cargarlo
checkpoint_path = './Checkpoints/mlp_checkpoint_latest_1000.pth'
start_epoch = 0
if os.path.isfile(checkpoint_path):
    print("Cargando checkpoint...")
    start_epoch = load_checkpoint(checkpoint_path)
    print(f"Resumir desde la época {start_epoch}")



import os

# el nuevo entrenanmiento ser 
checkpoint_path = './Checkpoints/mlp_checkpoint_latest_1000.pth'


def verify_dataset(image_paths):
    missing_files = [path for path in image_paths if not os.path.exists(path)]
    if missing_files:
        print(f"Archivos faltantes: {len(missing_files)}")
        for file in missing_files:
            print(file)
    else:
        print("Todos los archivos están presentes.")

# Llama a esta función antes de iniciar el entrenamiento
verify_dataset(dataset.image_paths)



# Inicializar lista para guardar las imágenes que no se identificaron
imagenes_no_identificadas = []

# Entrenamiento
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    # Barra de progreso para el dataloader
    for batch_idx, (images, labels) in enumerate(tqdm(subset_dataloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')):
        try:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, labels)

        except Exception as e:
            # Si ocurre un error, se salta el batch y se guarda la información
            print(f"Error al procesar el batch {batch_idx} de la epoch {epoch+1}: {e}")
            imagenes_no_identificadas.append((epoch+1, batch_idx))
            continue  # Salta este batch y continúa con el siguiente

    # Promediar la pérdida y la precisión para la época
    avg_loss = running_loss / len(subset_dataloader)
    avg_accuracy = epoch_accuracy / len(subset_dataloader)
    train_losses.append(avg_loss)

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint guardado en {checkpoint_path}")

    print(f"Epoch [{epoch+1}/{epochs}], Pérdida: {avg_loss:.4f}, Precisión: {avg_accuracy:.4f}")

print("Entrenamiento finalizado.")

# Mostrar imágenes no identificadas
if imagenes_no_identificadas:
    print("Imágenes no identificadas durante el entrenamiento:")
    for epoch_num, batch_num in imagenes_no_identificadas:
        print(f"Epoch {epoch_num}, Batch {batch_num}")

# Graficar la pérdida de entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Pérdida de Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el Entrenamiento')
plt.legend()
plt.show()
