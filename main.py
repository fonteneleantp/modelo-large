import torch
from ultralytics import YOLO

# Carregar o modelo YOLOv8
model = YOLO('yolov8n.pt')  # Substitua se necessário

# Caminho para o arquivo data.yaml
data_yaml_path = "data.yaml"

# Iniciando o treinamento com verbosidade e configuração para TensorBoard
results = model.train(
    data=data_yaml_path, 
    epochs=30, 
    imgsz=640, 
    batch=72, 
    verbose=True, 
    device=[0,1,2,3],
    project='runs',   # Diretório para salvar os logs
    name='exp',       # Nome da execução, será criado um subdiretório em 'runs'
)

# Exibindo os resultados do treinamento
print(results)