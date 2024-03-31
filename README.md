# Covid-19 X-Rays
![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/894edaf6-47b7-4997-bfc1-e3dfa8650615)

# Content
1. Introduction
2. Overview
3. Steps
4. Results
5. Usage
6. Conclusion
7. References

# Introduction
This study looks at utilizing artificial intelligence (AI) to rapidly and reliably detect COVID-19 in chest X-ray pictures. The authors created a huge dataset of chest X-rays, encompassing regular, viral pneumonia, and COVID-19 cases. They then used deep learning methods to build a system to evaluate X-rays and differentiate between these groups. The algorithm achieved extremely high accuracy, with success rates greater than 97% across all categories.  

Link: https://arxiv.org/abs/2003.13145

This project aims to make the former study a reality while dealing with the intricacies of the latter, with the help of Deep Learning.

# Overview 

Our project aimed to develop accurate models for COVID-19 detection using chest X-rays. We began by analyzing sample images showcasing positive COVID-19 cases. Following data exploration, we investigated the impact of image augmentation techniques on the dataset. To assess model performance, we built and compared four distinct prediction models: two convolutional neural networks (CNNs) built from scratch and two leveraging transfer learning. We then evaluated these models on their ability to accurately classify X-ray images. This involved analyzing which models achieved superior performance and identifying key hyperparameter settings that contributed to their success. We submitted the top three models to the COVID-19 Diagnostic AI Model Share competition to benchmark their performance against others. Following this initial submission, we continued our exploration, experimenting with additional models and comprehensively comparing their effectiveness. This in-depth analysis allowed us to definitively identify the best-performing models.

# Steps 
1. Data Exploration
2. Split the Dataset
3. Fine-tune VGG-16, ResNet-18 and DenseNet-121
  * *  I. Define Transformations
  * * II. Handle imbalanced dataset with Weighted Random Sampling (Over-sampling)
  *  III. Prepare the Pre-trained models
  * * IV. Fine-tune step with Early-stopping

5. Results Evaluation
  * I. Plot confusion matrices
    * II. Compute test-set Accuracy, Precision, Recall & F1-score
      * III. Localize using Grad-CAM
        * IV. Fine-tune step with Early-stopping

5. Inference 



# Results 



# Usage 

1. Clone the repository
2. Install dependencies
3. Using argparse script for inference
4. An example
   


# Conclusion 



# References 
