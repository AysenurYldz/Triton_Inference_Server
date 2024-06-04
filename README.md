# Triton Inference Server ile YOLOv8 Nesne Algılama

## Giriş

Bu rehber, önceden eğitilmiş bir YOLOv8 modelini kullanarak nesne algılama için Triton Inference Server'ı kurmanıza ve çalıştırmanıza yardımcı olacaktır. Bu, nesne algılama görevleri için ideal bir çözümdür ve Docker kapsayıcıları aracılığıyla dağıtımı kolaylaştırır.

## Gereklilikler

- Python 3.6 veya üstü
- NVIDIA GPU
- Docker
- Ultralytics YOLOv8 kod tabanı (https://github.com/ultralytics)
- TritonClient ([https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py](https://github.com/triton-inference-server))

## Dosya Yapısı

Bu rehberdeki dosyalar, nesne algılama için Triton Inference Server ve YOLOv8 modelini kullanmanızı kolaylaştırmak için organize edilmiştir:

### models/

Bu klasör, YOLOv8 modeli ve ilgili dosyaları içerir.

#### yolov8_onnx/
##### config.pbtxt
Birden fazla YOLOv8 modelini bir araya getiren bir ensemble modeli için yapılandırma dosyası.
##### 1/
- `model.onnx`: Bu dosya, YOLOv8 modelinin ONNX formatındaki temsilini içerir. Triton Inference Server bu dosyayı modeli yüklemek ve çalıştırmak için kullanır.
- `yolov8.pth`:  YOLOv8 modelinin PyTorch formatındaki temsili.

#### postprocess/
##### config.pbtxt
 Son işleme için özel yapılandırma dosyası.
##### 1/
- `model.py`: Bu dosya, YOLOv8 modelinin çıkışını işleyen ve son işleme uygulayan kodu içerir. Bu, algılanan nesneler için puan ve NMS (Maksimum Olmayan Bastırma) eşiklerini belirlemeyi içerir.

#### yolov8_ensemble/
##### config.pbtxt: 
Birden fazla YOLOv8 modelini bir araya getiren bir ensemble modeli için yapılandırma dosyası.

### predict.py

Bu ana komut dosyası, Triton Inference Server'ı başlatmak, modeli yüklemek ve bir görüntüden nesneleri algılamak için gerekli kodu içerir. Ayrıca algılanan nesneleri görselleştirmek veya kaydetmek için seçenekler de sunar.

---

Bu dosya yapısı, Triton Inference Server ve YOLOv8 modelini kullanarak nesne algılama için gerekli tüm bileşenleri organize ve erişilebilir hale getirir.

### Notlar:

- yolov8.pth ve config.pbtxt dosyaları, YOLOv8 modelini özelleştirmek için kullanılabilir. 
- model.py dosyası, özel gereksinimlerinize göre uyarlanabilir. Örneğin, algılanan nesneler için farklı görselleştirmeler ekleyebilirsiniz.
- predict.py dosyasında, görüntü dosya adı ve çıktı klasörü gibi parametreleri değiştirebilirsiniz

## Kurulum Adımları

### 1. Triton Server'ı İndirin ve Kurun

```bash
pip install triton-inference-server
```

### 2. Bağımlılıkları Kurun

```bash
pip install ultralytics==8.0.51 tritonclient[all]==2.31.0
```
### 3. (İsteğe bağlı) Son İşlemeyi ve Ensemble Yapılandırmasını Özelleştirin

- models/postprocess/1/model.py dosyasındaki puan ve NMS eşiklerini özel gereksinimlerinize göre ayarlayın.
- Farklı bir giriş çözünürlüğü kullanıyorsanız,

models/yolov8_ensemble/config.pbtxt dosyasını güncelleyin.

### 4. Docker Kapsayıcısını Oluşturun

```bash

DOCKER_NAME="yolov8-triton"
```

```bash
docker build -t $DOCKER_NAME .
```

### 5. Triton Inference Server'ı Çalıştırın

```bash
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
-it --rm \
--net=host \
-v ./models:/models \
$DOCKER_NAME
```

###6. Komutu Çalıştırın

```bash
python predict.py
```
## Kaynaklar

Triton Inference Server: https://developer.nvidia.com/triton-inference-server
YOLOv8: https://github.com/ultralytics
TritonClient: https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/http/init.py

## Notlar

predict.py dosyası, görüntü girdisini işlemek, nesneleri algılamak ve sonuçları görüntülemek veya kaydetmek için özel kod içerir. Bu dosyayı özel gereksinimlerinize göre uyarlamanız gerekebilir.

