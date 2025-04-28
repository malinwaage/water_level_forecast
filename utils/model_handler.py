import tensorflow as tf

def load_model(parameter):
    model_path = 'my_GRU_model_waterlevel.keras' if parameter == "1000" else 'my_GRU_model_discharge3.keras'
    return tf.keras.models.load_model(model_path)
