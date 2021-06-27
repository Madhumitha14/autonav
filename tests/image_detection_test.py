import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import glob
import shutil
import wget
import time
import numpy as np
import sys
sys.path.insert(1, "C:\\Users\\saile\\Desktop\\Sailesh\\College\\Mini Project\\")  # noqa
from environment import CarlaEnvironment  # noqa

CARLA_PATH = "C:\\Users\\saile\\Desktop\\Sailesh\\Carla Simulator\\CARLA_0.9.10\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-*%d.%d-%s.egg"  # noqa
try:
    sys.path.append(glob.glob(CARLA_PATH % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla  # noqa

os.chdir("C:\\Users\\saile\\Desktop\\Sailesh\\College\\Mini Project")

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 533

PATHS = {
    'APIMODEL_PATH': os.path.join('train', 'tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('train', 'tensorflow', 'annotations'),
    'IMAGE_PATH': os.path.join('train', 'images'),
    'MODEL_PATH': os.path.join('train', 'tensorflow', 'working_model'),
    'PRETRAINED_MODEL_PATH': os.path.join('train', 'tensorflow', 'pretrained models'),
    'CHECKPOINT_PATH': os.path.join('train', 'tensorflow', 'working_model', 'ssd_modnet'),
    'OUTPUT_PATH': os.path.join('train', 'tensorflow', 'working_model', 'ssd_modnet', 'export'),
    'TFJS_PATH': os.path.join('train', 'tensorflow', 'working_model', 'ssd_modnet', 'tfjsexport'),
    'TFLITE_PATH': os.path.join('train', 'tensorflow', 'working_model', 'ssd_modnet', 'tfliteexport'),
    'PROTOC_PATH': os.path.join('train', 'tensorflow', 'protoc-3.15.6-win64')
}

FILES = {
    'PIPELINE_CONFIG': os.path.join(PATHS['CHECKPOINT_PATH'], 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join('train', 'generate_tf_records.py'),
    'LABELMAP': os.path.join(PATHS['ANNOTATION_PATH'], 'label_map.pbtxt')
}


class CollectImages:
    def __init__(self):
        self.labels = ["car", "lamp", "pedestrian", "cyclists", "street_light"]
        self.images_path = os.path.join("train", "images", "collected_images")
        self.dump_path = os.path.join(
            "train", "images", "image_dump", "object_detection")
        self.num_images = 10
        self.create_file_paths()

    def create_file_paths(self):
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        else:
            shutil.rmtree(self.dump_path, ignore_errors=True)
            os.mkdir(self.dump_path)
        for label in self.labels:
            path = os.path.join(self.images_path, label)
            if not os.path.exists(path):
                os.mkdir(f"{self.images_path}\\{label}")

    def collect_images(self, image):
        img = np.array(image.raw_data)
        img = img.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        img = img[:, :, :3]
        cv2.imwrite(f"{self.dump_path}\\{int(time.time())}.jpeg", img)
        cv2.waitKey(1)

    def carla_connect(self):
        client = CarlaEnvironment(IMAGE_HEIGHT, IMAGE_WIDTH)
        vehicle_blueprint = client.blueprint_library.filter("vehicle.audi.a2")[0]  # noqa
        vehicle = client.world.try_spawn_actor(
            vehicle_blueprint, random.choice(client.spawn_points))
        if vehicle == None:
            print("unable to spawn vehicle")
            return None
        vehicle.set_autopilot(enabled=True)
        client.actor_list.append(vehicle)
        rgb_camera_bp = client.blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute("image_size_x", f"{IMAGE_WIDTH}")
        rgb_camera_bp.set_attribute("image_size_y", f"{IMAGE_HEIGHT}")
        rgb_camera_bp.set_attribute("fov", '120')
        rgb_camera = client.world.try_spawn_actor(rgb_camera_bp, carla.Transform(carla.Location(3, 0, 2)), attach_to=vehicle)  # noqa
        client.actor_list.append(rgb_camera)
        # rgb_camera.listen(lambda image: self.collect_images(image))
        return vehicle, client, rgb_camera


def collect():
    start_time = time.time()
    image_collector = CollectImages()
    vehicle, client, camera = image_collector.carla_connect()
    if vehicle == None:
        return
    camera.listen(lambda image: image_collector.collect_images(image))
    while True:
        if time.time() - start_time > 600:
            break
    client.cleanup()


def train_config():
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format

    config = config_util.get_configs_from_pipeline_file("train\\tensorflow\\working_model\\ssd_modnet\\pipeline.config")  # noqa

    labels = [{
        "name": 'car',
        "id": 1
    }, {
        "name": 'lamp',
        "id": 2
    }, {
        "name": 'pedestrian',
        "id": 3
    }, {
        "name": 'cyclist',
        "id": 4
    }, {
        "name": 'street_light',
        "id": 5
    }]

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(FILES['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 8
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PATHS['PRETRAINED_MODEL_PATH'], model_name, 'checkpoint', 'ckpt-0')  # noqa
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = FILES['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(PATHS['ANNOTATION_PATH'], 'train.record')]  # noqa
    pipeline_config.eval_input_reader[0].label_map_path = FILES['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(PATHS['ANNOTATION_PATH'], 'test.record')]  # noqa

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(FILES['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)


def train():
    train_config()
    TRAINING_SCRIPT = os.path.join(PATHS['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')  # noqa
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, PATHS['CHECKPOINT_PATH'], FILES['PIPELINE_CONFIG'])  # noqa
    os.system(command)


def evaluate():
    TRAINING_SCRIPT = os.path.join(PATHS['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')  # noqa
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, PATHS['CHECKPOINT_PATH'], FILES['PIPELINE_CONFIG'], PATHS['CHECKPOINT_PATH'])  # noqa
    os.system(command)


def test():
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util

    configs = config_util.get_configs_from_pipeline_file(FILES['PIPELINE_CONFIG'])  # noqa
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)  # noqa

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATHS['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()  # noqa

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(FILES['LABELMAP'])  # noqa
    IMAGE_PATH = os.path.join(PATHS['IMAGE_PATH'], 'test', '1624730085.jpeg')
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}  # noqa
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)  # noqa

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    cv2.imshow("box", cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))  # noqa
    cv2.waitKey(0)


if __name__ == "__main__":
    model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
    pretrained_model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
    # collect()
    # train()
    evaluate()
    # test()
