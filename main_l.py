import torch
from ultralytics import YOLO

# Carregar o modelo YOLOv8
model = YOLO('yolov8l.pt')  # Substitua se necessário

# Caminho para o arquivo data.yaml
data_yaml_path = "data.yaml"

# Iniciando o treinamento com verbosidade e configuração para TensorBoard
results = model.train(
    data=data_yaml_path, 
    epochs=30, 
    imgsz=640, 
    batch=16, 
    verbose=True, 
    device=[0,1,2,3],
    project='runs',   # Diretório para salvar os logs
    name='exp_l',       # Nome da execução, será criado um subdiretório em 'runs'
)

# Exibindo os resultados do treinamento
print(results)