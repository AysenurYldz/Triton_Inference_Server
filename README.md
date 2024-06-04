# Triton Inference Server ile YOLOv8 Nesne Algılama

## Giriş

Bu rehber, önceden eğitilmiş bir YOLOv8 modelini kullanarak nesne algılama için Triton Inference Server'ı kurmanıza ve çalıştırmanıza yardımcı olacaktır. Bu, nesne algılama görevleri için ideal bir çözümdür ve Docker kapsayıcıları aracılığıyla dağıtımı kolaylaştırır.

## Gereklilikler

Python 3.6 veya üstü
NVIDIA GPU (isteğe bağlı, CPU'da da çalışabilir)
Docker
Ultralytics YOLOv8 kod tabanı (https://github.com/ultralytics)
TritonClient (https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py)

## Dosya Yapısı
Bu rehberdeki dosyalar, nesne algılama için Triton Inference Server ve YOLOv8 modelini kullanmanızı kolaylaştırmak için organize edilmiştir:

models/:

Bu klasör, YOLOv8 modeli ve ilgili dosyaları içerir.

yolov8_onnx/:

1/:
model.onnx: Bu dosya, YOLOv8 modelinin ONNX formatındaki temsilini içerir. Triton Inference Server bu dosyayı modeli yüklemek ve çalıştırmak için kullanır.
postprocess/:

1/:
model.py: Bu dosya, YOLOv8 modelinin çıkışını işleyen ve son işleme uygulayan kodu içerir. Bu, algılanan nesneler için puan ve NMS (Maksimum Olmayan Bastırma) eşiklerini belirlemeyi içerir.
yolov8_ensemble/:

config.pbtxt: Bu dosya (isteğe bağlı), birden fazla YOLOv8 modelini bir araya getiren bir ensemble modeli için yapılandırmayı tanımlar.
predict.py:

Bu ana komut dosyası, Triton Inference Server'ı başlatmak, modeli yüklemek ve bir görüntüden nesneleri algılamak için gerekli kod içerir. Ayrıca algılanan nesneleri görselleştirmek veya kaydetmek için seçenekler de sunar.

Bu dosya yapısı, Triton Inference Server ve YOLOv8 modelini kullanarak nesne algılama için gerekli tüm bileşenleri organize ve erişilebilir hale getirir.

Dosyalara Dahil Olanlar:

model.onnx: YOLOv8 modelinin ONNX formatındaki temsili
model.py: YOLOv8 modelinin çıkışını işleyen ve son işleme uygulayan kod
config.pbtxt: (isteğe bağlı) Birden fazla YOLOv8 modelini bir araya getiren bir ensemble modeli için yapılandırma
predict.py: Triton Inference Server'ı başlatan, modeli yükleyen ve görüntüden nesneleri algılayan ana komut dosyası

## Kurulum Adımları

1. Triton Server'ı İndirin ve Kurun:
Bash
pip install triton-inference-server
Kodu dikkatli kullanın.
content_copy
2. Bağımlılıkları Kurun:
Bash
pip install ultralytics==8.0.51 tritonclient[all]==2.31.0
Kodu dikkatli kullanın.
content_copy
3. (İsteğe bağlı) Son İşlemeyi ve Ensemble Yapılandırmasını Özelleştirin:
models/postprocess/1/model.py dosyasındaki puan ve NMS eşiklerini özel gereksinimlerinize göre ayarlayın.
Farklı bir giriş çözünürlüğü kullanıyorsanız, models/yolov8_ensemble/config.pbtxt dosyasını güncelleyin.
4. Docker Kapsayıcısını Oluşturun:
Bash
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
Kodu dikkatli kullanın.
content_copy
5. Triton Inference Server'ı Çalıştırın:
Bash
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
  -it --rm \
  --net=host \
  -v ./models:/models \
  $DOCKER_NAME
Kodu dikkatli kullanın.
content_copy
6. Komutu Çalıştırın:
Bash
python predict.py
Kodu dikkatli kullanın.
content_copy

## Kaynaklar

Triton Inference Server: https://developer.nvidia.com/triton-inference-server
YOLOv8: https://github.com/ultralytics
TritonClient: https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py

## Notlar

predict.py dosyası, görüntü girdisini işlemek, nesneleri algılamak ve sonuçları görüntülemek veya kaydetmek için özel kod içerir. Bu dosyayı özel gereksinimlerinize göre uyarlamanız gerekebilir.

