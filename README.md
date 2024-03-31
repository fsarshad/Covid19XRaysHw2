# Covid-19 X-Rays

# Content
1. Introduction
2. Overview
3. Steps
4. Results
5. Usage
6. Conclusion
7. References

# Introduction
This study looks at utilizing artificial intelligence (AI) to rapidly and reliably detect COVID-19 in chest X-ray pictures. The authors created a huge dataset of chest X-rays, encompassing regular, viral pneumonia, and COVID-19 cases. They then used deep learning methods to build a system to evaluate X-rays and differentiate between these groups. The algorithm achieved extremely high accuracy, with success rates greater than 97% across all categories [https://arxiv.org/abs/2003.13145](#1)

.
This project aims to make the former study a reality while dealing with the intricacies of the latter, with the help of Deep Learning.

# Overview 


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
