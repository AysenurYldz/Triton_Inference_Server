import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import sys
import argparse

# Classes names:
classes_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Renkler
class_colors = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0),
    8: (0, 0, 128), 9: (128, 128, 0), 10: (128, 0, 128), 11: (0, 128, 128),
    12: (255, 128, 0), 13: (255, 0, 128), 14: (128, 255, 0), 15: (0, 128, 255),
    16: (255, 128, 128), 17: (128, 255, 128), 18: (128, 128, 255),
    19: (255, 255, 128), 20: (255, 128, 255), 21: (128, 255, 255),
    22: (192, 192, 192), 23: (128, 128, 128), 24: (0, 0, 0)
}


def get_triton_client(url: str = 'localhost:9100'):
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def draw_bounding_box(img, class_name, confidence, x, y, x_plus_w, y_plus_h):
    color = class_colors[hash(class_name) % len(class_colors)]
    label = f'{class_name} ({confidence:.2f})'
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def read_image(image_path: str, expected_image_shape) -> np.ndarray:
    expected_width = expected_image_shape[0]
    expected_height = expected_image_shape[1]
    expected_length = min((expected_height, expected_width))
    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / expected_length

    input_image = cv2.resize(image, (expected_width, expected_height))
    input_image = (input_image / 255.0).astype(np.float32)

    # Channel first
    input_image = input_image.transpose(2, 0, 1)

    # Expand dimensions
    input_image = np.expand_dims(input_image, axis=0)
    return original_image, input_image, scale


def run_inference(model_name: str, input_image: np.ndarray,
                  triton_client: grpcclient.InferenceServerClient):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput('num_detections'))
    outputs.append(grpcclient.InferRequestedOutput('detection_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('detection_scores'))
    outputs.append(grpcclient.InferRequestedOutput('detection_classes'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    num_detections = results.as_numpy('num_detections')
    detection_boxes = results.as_numpy('detection_boxes')
    detection_scores = results.as_numpy('detection_scores')
    detection_classes = results.as_numpy('detection_classes')
    return num_detections, detection_boxes, detection_scores, detection_classes


def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    expected_image_shape = triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
    original_image, input_image, scale = read_image(image_path, expected_image_shape)
    num_detections, detection_boxes, detection_scores, detection_classes = run_inference(
        model_name, input_image, triton_client)

    for index in range(num_detections):
        box = detection_boxes[index]
        class_index = int(detection_classes[index])
        class_name = classes_names[class_index]
        draw_bounding_box(original_image,
                          class_name,
                          detection_scores[index],
                          round(box[0] * scale),
                          round(box[1] * scale),
                          round((box[0] + box[2]) * scale),
                          round((box[1] + box[3]) * scale))

    cv2.imwrite('output.jpg', original_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='assets/dog_and_girl.jpg')
    parser.add_argument('--model_name', type=str, default='yolov8_ensemble')
    parser.add_argument('--url', type=str, default='localhost:8001')
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)

