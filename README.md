# Skin Lesion Classification using Class Expert Models and GANs

## Project Overview
This notebook explores the development of a deep learning model for classifying skin lesions from the HAM10000 dataset. A key focus is addressing the severe class imbalance inherent in medical image datasets, particularly by leveraging Generative Adversarial Networks (GANs) to create 'class expert' features and integrating them into a Convolutional Neural Network (CNN) for improved classification.

## Dataset
The project utilizes the **HAM10000 (Human Against Machine with 10000 training images)** dataset, which contains 10,015 dermatoscopic images of common pigmented skin lesions. The dataset is notably imbalanced, with a disproportionate number of benign cases compared to malignant ones and other rarer lesion types.

## Approach
1.  **Data Loading and Preprocessing**: Images and metadata are loaded, and initial exploratory data analysis is performed to understand distributions of gender, age, and lesion types.
2.  **Image Organization**: Images are organized into class-specific directories based on their diagnosis to facilitate class-wise processing.
3.  **GAN-based Class Experts**: Separate GANs are trained for each lesion class. The discriminators from these GANs are intended to act as 'class expert' feature extractors, learning to differentiate real images of a specific class from generated fakes.
4.  **Integrated CNN Classifier**: A main CNN model is constructed, incorporating the feature extraction capabilities of these 'class expert' discriminators. This architecture aims to benefit from the specialized knowledge of each class expert.
5.  **Robust Training and Evaluation**: The model training incorporates advanced techniques:
    *   **Data Augmentation**: To enhance generalization and mitigate overfitting.
    *   **Regularization**: L2 regularization and Dropout layers are used to prevent overfitting.
    *   **Learning Rate Scheduling**: To optimize the learning process over epochs.
    *   **Early Stopping**: To prevent overfitting and save the best model weights.
    *   **K-Fold Cross-Validation**: To ensure robust evaluation of the model's performance across different data splits.
    *   **Bayesian Model Averaging (BMA)**: Applied to combine predictions from class experts for a potentially more robust final prediction.

## Technologies and Libraries
*   **Python**
*   **TensorFlow/Keras**: For building and training deep learning models (GANs and CNNs).
*   **Pandas**: For data manipulation and analysis.
*   **Numpy**: For numerical operations.
*   **Matplotlib & Seaborn**: For data visualization.
*   **Scikit-learn**: For data splitting and classification metrics.

## Notebook Structure
-   **Data Loading and EDA**: Initial setup, data mounting, unzipping, and preliminary analysis.
-   **Image Path Mapping**: Linking image files to DataFrame records.
-   **GAN Training Setup**: Defining Generator and Discriminator models for class-specific training.
-   **Class Expert CNN Development**: Constructing the main classification model that integrates the 'class expert' concept.
-   **Training and Evaluation with K-Fold**: Implementing cross-validation, training callbacks, and detailed performance reporting.

