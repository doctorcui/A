import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# sys.path.append('../..')
import tensorflow as tf
from tensorflow.keras import layers

#####pb 文件  转TFLITE文件
def convert_to_tflite(pb_path,lite_path,model_tag,step):
    for quant_type in "Int8,Float32".split(","):
        model = tf.keras.models.load_model(pb_path)
        model.inputs[0].set_shape((1,8,229,1))
        # model.outputs[0].set_shape([tflite_input_length, midi_pitches])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quant_type == "Int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quant_type == "Float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quant_type == "Float32":
            converter.optimizations = []
        tflite_model = converter.convert()
        with open(f'{lite_path}/quant_new_{quant_type}_{model_tag}_{step}.tflite', "wb") as f:
            f.write(tflite_model)
###tf-lite主要是针对移动端进行优化的平台，重新定义了移动端的核心算子，
# 也提供了硬件加速的接口，拥有新的优化解释器。
from tqdm import tqdm
if __name__ == "__main__":
    model_tag = "base_v2"
    for step in tqdm([str(i).zfill(5) for i in [50000,45000,55000,65000,60000]]):
        pb_path = f"/data/models/{model_tag}/model_files/{step}"
        lite_path = f"/data/models/{model_tag}/model_files/{step}"
        convert_to_tflite(pb_path, lite_path,model_tag,step).
        