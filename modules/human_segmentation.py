import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter

# Инициализация устройства
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Функция для создания модели
def init_segmentation_frames(num_classes: int = 1, aux_loss: bool = False):
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, dilated=True).features

    # Выходной слой, соответствующий глубокому представлению
    return_layers = {'12': 'out'}
    if aux_loss:
        return_layers['9'] = 'aux'

    # Получаем выходные признаки из нужного слоя MobileNetV3
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # Определяем классификатор
    classifier = DeepLabHead(576, num_classes)
    aux_classifier = FCNHead(96, num_classes) if aux_loss else None

    # Создаем модель DeepLabV3
    model = DeepLabV3(backbone, classifier, aux_classifier)

    return model

# Загрузка модели
model = init_segmentation_frames().to(DEVICE)
model.load_state_dict(torch.load('models/people_segmentation_mobilenet_v3_small.pth'))
model.eval()  # Переводим модель в режим инференса

# Преобразование изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для обработки каждого кадра
def process_frame(frame, model):
    # Применяем преобразования
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(DEVICE)  # Добавляем размер батча (1, C, H, W)

    # Получаем предсказания
    with torch.no_grad():
        output = model(image)['out']  # Получаем выход для всего батча
        output = output[0]  # Берем первое изображение из батча
        output = torch.sigmoid(output)  # Применяем сигмоиду для бинарной маски
        output = output.cpu().numpy()

    # Преобразуем маску обратно в изображение
    mask = output[0]  # Берем первый канал (если num_classes = 1)
    mask = np.array(mask > 0.5, dtype=np.uint8) * 255  # Преобразуем в 0 и 255 для визуализации
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask

# Генерация кадров для видеопотока
def generate_human_segmentation_frames(cap, model):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получаем маску для распознавания людей
        mask = process_frame(frame, model)

        # Преобразуем кадр и маску в формат JPEG
        combined_frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

        # Преобразуем в JPEG
        _, jpeg_frame = cv2.imencode('.jpg', combined_frame)
        frame = jpeg_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')