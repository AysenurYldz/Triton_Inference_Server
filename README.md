# Giriş

Bu rehber, önceden eğitilmiş YOLOv8 modelini kullanarak nesne algılama için Triton Inference Server'ı kurmanıza ve çalıştırmanıza yardımcı olacaktır. Bu, nesne algılama görevleri için ideal bir çözümdür ve Docker kapsayıcıları aracılığıyla dağıtımı kolaylaştırır.

# Gereklilikler

Python 3.6 veya üstü
NVIDIA GPU (isteğe bağlı, CPU'da da çalışabilir)
Docker
Ultralytics YOLOv8 kod tabanı (https://github.com/ultralytics)
TritonClient (https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py)

# Dosya Yapısı

Bu rehberdeki dosyalar aşağıdaki gibi organize edilmiştir:

models/:

yolov8_onnx/:

1/:

model.onnx: ONNX formatındaki YOLOv8 modeli

postprocess/:

1/:

model.py: Son işleme için kod (puan ve NMS eşikleri)

yolov8_ensemble/:

config.pbtxt: Ensemble yapılandırma dosyası 

predict.py: Triton Inference Server ve YOLOv8 modelini kullanarak nesne algılama için ana komut dosyası

# Kurulum Adımları

## Bağımlılıkları Kurun:

Bash
pip install ultralytics==8.0.51 tritonclient[all]==2.31.0
Kodu dikkatli kullanın.
content_copy
(İsteğe bağlı) Son İşlemeyi ve Ensemble Yapılandırmasını Özelleştirin:

models/postprocess/1/model.py dosyasındaki puan ve NMS eşiklerini özel gereksinimlerinize göre ayarlayın.
Farklı bir giriş çözünürlüğü kullanıyorsanız, models/yolov8_ensemble/config.pbtxt dosyasını güncelleyin.
Docker Kapsayıcısını Oluşturun:

Bash
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
Kodu dikkatli kullanın.
content_copy
Triton Inference Server'ı Çalıştırın:

Bash
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
  -it --rm \
  --net=host \
  -v ./models:/models \
  $DOCKER_NAME
Kodu dikkatli kullanın.
content_copy
Komutu Çalıştırın:

Bash
python main.py
Kodu dikkatli kullanın.
content_copy


Kaynaklar
Triton Inference Server: https://developer.nvidia.com/triton-inference-server
YOLOv8: https://github.com/ultralytics
TritonClient: https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py
