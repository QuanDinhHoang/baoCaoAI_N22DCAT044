# baoCaoAI_N22DCAT044

My Project : Face classification using Convolutional Neural Networks (CNN) on the LFW dataset

# 1. LFW Dataset (Labeled Faces in the Wild)

The Labeled Faces in the Wild (LFW) dataset is a facial image database developed to explore the problem of unconstrained face recognition.

- **Purpose:** Released for research to advance face verification, not for commercial algorithm evaluation.
- **Goal:** Evaluate the accuracy of face recognition systems under uncontrolled real-world conditions (in-the-wild).

## 🔑 Key characteristics of the LFW dataset:

- 📸 **Number of images:** 13,233 facial images  
- 👥 **Number of individuals:** 5,749 different people  
- 🔁 **Images per individual:** Uneven; ~1,680 people have >1 image  
- 🌐 **Image sources:** Collected from the Internet (mainly news websites)  
- 🏷️ **Labeling:** Each image is labeled with the person’s name  
- 🖼️ **Image size:** 250×250 pixels (DeepFunneled version aligned & normalized)



# 2. Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)**, also known as *ConvNet*, is a specialized type of deep learning algorithm primarily designed for object recognition tasks such as image classification. CNNs are widely used in real-world applications such as self-driving cars, security camera systems, and many other domains.

## 🔍 Why CNNs Are Important

CNNs have become essential in modern AI due to several key advantages:

- 🚀 **Superior performance** over traditional machine learning algorithms (e.g., SVM, decision trees) by automatically extracting large-scale features — no manual feature engineering needed.
- 🧠 **Translation invariance**: Convolutional layers allow CNNs to detect and extract patterns regardless of changes in position, orientation, scale, or displacement.
- 🏗️ **Pre-trained architectures** like `VGG-16`, `ResNet50`, `Inceptionv3`, and `EfficientNet` offer strong performance and can be fine-tuned for new tasks with limited data.
- 🔄 **Versatility**: CNNs go beyond image classification — they’re also used in:
  - 📈 Time series analysis  
  - 🗣️ Speech recognition  
  - 📝 Natural Language Processing (NLP)

## 🧩 Main Components of a CNN

A CNN typically consists of four key components:

1. **Convolutional Layers** – for automatic feature extraction.  
2. **Rectified Linear Units (ReLU)** – for introducing non-linearity.  
3. **Pooling Layers** – commonly MaxPooling, used to downsample feature maps.  
4. **Fully Connected Layers** – for final classification based on extracted features.


# 3. About My Project

## 📁 Dataset Preparation – LFW (Labeled Faces in the Wild)

- ✅ **Step 1**: The program iterates through all subdirectories in the LFW dataset, counting the number of images per individual.
- ✅ **Step 2**: Only individuals with **≥ 30 images** are selected to ensure a sufficiently large dataset and to avoid training bias.
- ✅ **Step 3**: Using the `face_recognition` library, the program detects and extracts face regions from the selected images.
- ✅ **Step 4**: Each face image is resized to a standard size of **100×100 pixels**.
- ✅ **Step 5**: Corresponding labels are encoded into numeric format using `LabelEncoder`.
- ✅ **Step 6**: The dataset is split into:
  - **Training set**: 80%  
  - **Validation set**: 10%  
  - **Test set**: 10%  
  using **stratified sampling** to preserve class distribution across all subsets.

---

## 🧠 Model Architecture & Training

- 🧩 **Model Structure**:
  - Built using **4 blocks** of:
    - `Conv2D`
    - `BatchNormalization`
    - `MaxPooling2D`
  - Followed by **2 Dense layers** with a `Dropout` of **0.6** to prevent overfitting.

- ⚙️ **Training Configuration**:
  - **Loss function**: `categorical_crossentropy`
  - **Optimizer**: `Adam`
  - **Epochs**: 60
  - **Batch size**: 64
  - **Validation**: Model performance is monitored on the validation set during training.


## 🚀4. Cài đặt & chạy thử
-Clone the project into the D drive because I have set up the program to only load the LFW dataset from a directory with an absolute path :
git clone https://github.com/QuanDinhHoang/baoCaoAI_N22DCAT044.git
-Download the appropriate version of Anaconda for your operating system using the link: https://www.anaconda.com/download/success
-Then create a new virtual environment on Anaconda and install some libraries on that virtual environment:
absl-py==2.2.2
altair==5.5.0
astunparse==1.6.3
attrs==25.3.0
beautifulsoup4==4.13.4
blinker==1.9.0
cachetools==5.5.2
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.1.8
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
deepface==0.0.93
dlib @ file:///D:/bld/dlib-split_1745321986600/work
face-recognition==1.3.0
face_recognition_models==0.3.0
filelock==3.18.0
fire==0.7.0
Flask==3.1.0
flask-cors==5.0.1
flatbuffers==25.2.10
fonttools==4.57.0
gast==0.6.0
gdown==5.2.0
gitdb==4.0.12
GitPython==3.1.44
google-pasta==0.2.0
grpcio==1.71.0
gunicorn==23.0.0
h5py==3.13.0
idna==3.10
imageio==2.37.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.0
jsonschema==4.23.0
jsonschema-specifications==2025.4.1
keras==3.9.2
kiwisolver==1.4.8
lazy_loader==0.4
libclang==18.1.1
lz4==4.4.4
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.1
mdurl==0.1.2
ml_dtypes==0.5.1
mtcnn==1.0.0
namex==0.0.9
narwhals==1.37.1
networkx==3.4.2
numpy==2.1.3
opencv-python==4.11.0.86
opt_einsum==3.4.0
optree==0.15.0
packaging==24.2
pandas==2.2.3
piexif==1.1.3
pillow==11.2.1
protobuf==5.29.4
pyarrow==20.0.0
pydeck==0.9.1
Pygments==2.19.1
pyparsing==3.2.3
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.2
referencing==0.36.2
requests==2.32.3
retina-face==0.0.17
rich==14.0.0
rpds-py==0.24.0
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.15.3
six==1.17.0
smmap==5.0.2
soupsieve==2.7
streamlit==1.45.0
tenacity==9.1.2
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensorflow==2.19.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==3.1.0
tf_keras==2.19.0
threadpoolctl==3.6.0
tifffile==2025.3.30
toml==0.10.2
tornado==6.4.2
tqdm==4.67.1
typing_extensions==4.13.2
tzdata==2025.2
urllib3==2.4.0
watchdog==6.0.0
Werkzeug==3.1.3
wrapt==1.17.2
XlsxWriter==3.2.3
-Change the path in DATASET_PATH which is from train.py:
DATASET_PATH = r"D:\baoCaoAI_N22DCAT044\archive\lfw-deepfunneled\lfw-deepfunneled"
-Run train.py within the activated Anaconda environment :
python D:\baoCaoAI_N22DCAT044\train.py

