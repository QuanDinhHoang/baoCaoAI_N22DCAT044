import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

lfw_cnn_model = "/mnt/d/project/lfw_cnn_model.h5"
label_encoder = "/mnt/d/project/label_encoder.pkl"
X_test_final = "/mnt/d/project/X_test_final.npy"
Y_test_cat = "/mnt/d/project/y_test_cat.npy"

# Load model và label encoder
model = load_model(lfw_cnn_model)
with open(label_encoder, "rb") as f:
    label_encoder = pickle.load(f)

# Load dữ liệu test đã lưu sẵn
X_test_final = np.load(X_test_final)
y_test_cat = np.load(Y_test_cat)

print("\n--- Hiển thị toàn bộ ảnh test cùng nhãn thật và dự đoán ---")
step = 10
for start in range(0, len(X_test_final), step):
    end = min(start + step, len(X_test_final))
    num_images = end - start
    cols = 5
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))

    for i in range(start, end):
        img = X_test_final[i]
        img_input = np.expand_dims(img, axis=0)

        true_label_index = np.argmax(y_test_cat[i])
        true_label = label_encoder.inverse_transform([true_label_index])[0]

        prediction = model.predict(img_input, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        plt.subplot(rows, cols, i - start + 1)
        plt.imshow(img)
        plt.title(f"T:{true_label}\nĐ:{predicted_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
