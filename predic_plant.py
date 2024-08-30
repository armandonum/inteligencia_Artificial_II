import os
import torch
from torchvision import transforms
from PIL import Image
from mlp import MLP

# Verificar si se dispone de una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapeo de etiquetas numéricas a nombres de clases
class_map = {
    0: 'achira',
    1: 'calanchoe',
    2: 'diente_de_leon',
    3: 'flor_de_rosa',
    4: 'palta'
}

# Definir las transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el modelo entrenado
model = MLP().to(device)
checkpoint_path = './Checkpoints/mlp_checkpoint_latest_500.pth'

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado desde {checkpoint_path}")

load_checkpoint(checkpoint_path)
model.eval()

# Función para predecir la clase de una imagen
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Transformar y añadir un batch dimension
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            class_name = class_map[predicted.item()]
            return class_name
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

if __name__ == "__main__":
    # Ruta de la imagen a clasificar
    image_path = './imagenes_prueba/image.png'

    if os.path.exists(image_path):
        class_name = predict_image(image_path)
        if class_name:
            print(f"La imagen pertenece a la clase: {class_name}")
        else:
            print("No se pudo predecir la clase de la imagen.")
    else:
        print(f"El archivo {image_path} no existe.")
