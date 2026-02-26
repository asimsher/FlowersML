from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
import uvicorn
import torch.nn as nn
from PIL import Image
import io
import streamlit as st


classes = [
    'daisy',
    'dandelion',
    'rose',
    'sunflower',
    'tulip'
]


class Flowers_1(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d( 128 ),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8)),
        )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x


transform_a = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Flowers_1()
model.load_state_dict(torch.load('model_flowers.pth', map_location=device))
model.to(device)
model.eval()


st.title('Flowers Project ML')
st.text('Загрузите изображение с цветок, и модел попробуе ее распознат')

flowers_img = st.file_uploader('Выберите изображени', type=['png', 'jpg', 'jpeg', 'svg'])
if not flowers_img:
    st.info('Upload Image')
else:
    st.image(flowers_img, caption='Uploaded image')

if st.button('Опредилите цветы'):
    try:
        image_data = flowers_img.read()
        img = Image.open( io.BytesIO( image_data ) )
        img_tensor = transform_a( img ).unsqueeze( 0 ).to( device )

        with torch.no_grad():
            pred = model( img_tensor )
            result = pred.argmax( dim=1 ).item()
        st.success(f'Модель думает, что это цветы: {classes[result]}')

    except Exception as e:
        st.exception(f'Error {str(e)}')
