import streamlit as st
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch 
import torch.nn as nn
import os
import torchvision 

# def labels():
#     root_dir = '/home/phan/final_ai/Vehicles'
#     classes = {
#     label_idx: class_name \
#         for label_idx, class_name in enumerate(
#             sorted(os. listdir(root_dir))
#         )
#     }
#     return classes

# classes = labels()
classes = {
            0:"Auto Rickshaws",
            1:"Bikes",
            2:"Cars",
            3:"Motorcycles",
            4:"Planes",
            5:"Ships",
            6:"Trains",
            }

st.title('Chào mừng bạn đến với phần mềm phân loại phương tiện giao thông')
st.header('Điều bạn cần làm là cung cấp 1 bức ảnh cho chúng tôi')

resnet18 = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
feature = nn.Sequential(*list(resnet18.children())[:-2])

model = nn.Sequential(
    feature,
    nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(512*7*7,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,len(classes))
)



model.load_state_dict(torch.load('resnet18_weights.pth', map_location=torch.device('cpu')))
model.eval()






st.markdown("<h3>Chọn FILE</h3>", unsafe_allow_html=True)
image = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
if image is  not None:
   
    image = Image.open(image)
    image = image.convert('RGB')
    st.image(image, caption="Ảnh cần dự đoán")
    
    
    test_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                        ])
    
    image_tensor = test_transform(image)

    if st.button('Dự đoán'):
        with torch.no_grad():
            batch = torch.unsqueeze(image_tensor,0)
            out = model(batch)

        _, predicted = torch.max(out, 1)
        index = predicted.item()
        result = classes[index]
        st.markdown(f"<h2>Kết quả dự đoán: <span style='color: green;'>{result}</span></h2>", unsafe_allow_html=True)



