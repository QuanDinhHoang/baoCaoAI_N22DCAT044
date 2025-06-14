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


## 🚀 4. Installation & Execution

### 🔹 Clone the Project

> ⚠️ The program is configured to load the LFW dataset using an absolute path, so please clone the project into the **D drive**:

```bash
git clone https://github.com/QuanDinhHoang/baoCaoAI_N22DCAT044.git

Set Up the Environment
Download and install Anaconda from the official website:
👉 https://www.anaconda.com/download/success

Create a new virtual environment (recommended: Python 3.10):

conda create -n lfw_env python=3.10
conda activate lfw_env
📦 Install Required Libraries
Install all required Python libraries using pip:

pip install -r requirements.txt
If requirements.txt is not available, manually install the libraries listed in the documentation.

📁 Set the Dataset Path
In the train.py file, update the DATASET_PATH to match your local dataset directory:


DATASET_PATH = r"D:\baoCaoAI_N22DCAT044\archive\lfw-deepfunneled\lfw-deepfunneled"
⚠️ Ensure the dataset is extracted to this path or update it based on your dataset's actual location.

▶️ Run the Training Script
Once everything is set up, activate your environment and execute the training file:
python D:\baoCaoAI_N22DCAT044\train.py











