# SVM Image Classification for Dog and Cat Recognition

## Quickstart
- **Dependencies**: Python, scikit-learn, OpenCV, NumPy, pandas
- **Installation**: Run `pip install scikit-learn opencv-python numpy pandas`
- **Usage**: Execute `t3.py` to classify images of dogs and cats.

## Overview
This project utilizes a Support Vector Machine (SVM) classifier to distinguish between images of dogs and cats with high accuracy. The classifier undergoes training on a grayscale image dataset and determines the category of each image.

## Data
The dataset used for this project is the "Dogs vs. Cats" dataset, which is available on Kaggle. To access and download the dataset, please click [here](https://www.kaggle.com/c/dogs-vs-cats/data).




## Results
- **Accuracy**: The model's accuracy on the test dataset is quantified as a percentage.
- **Odds of breaking the Asirra CAPTCHA**: This metric calculates the probability of consecutively classifying 12 images correctly, based on the model's accuracy.
- `t3submission.csv`: This CSV file contains the model's predicted labels for the test dataset.

## Contribute
Contributions aimed at enhancing the classifier's accuracy or refining the preprocessing steps are highly encouraged. Please fork the repository and create a pull request with your proposed modifications.

## License
This project is distributed under the MIT License. See the LICENSE file in the repository for more information.
