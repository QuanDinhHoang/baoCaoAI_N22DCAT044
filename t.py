from tensorflow.keras.models import load_model

model = load_model("D:/project/lfw_cnn_model.h5")
model.save("D:/project/lfw_model_saved")


