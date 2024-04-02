# Covid-19 Chest X-Rays Deep Learning 
![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/894edaf6-47b7-4997-bfc1-e3dfa8650615)

# Content
1. Introduction
2. Overview
3. Steps
4. Results
5. Conclusion
6. References

# Introduction
The study below looks at utilizing artificial intelligence (AI) to rapidly and reliably detect COVID-19 in chest X-ray pictures. The authors created a huge dataset of chest X-rays, encompassing regular, viral pneumonia, and COVID-19 cases. They then used deep learning methods to build a system to evaluate X-rays and differentiate between these groups. 

Link: https://arxiv.org/abs/2003.13145

As a team of three, at Columbia University, our project will use deep learning to try and produce similar results. 

# Overview 

Our project aimed to develop accurate models for COVID-19 detection using chest X-rays. We began by analyzing sample images showcasing positive COVID-19 cases. Following data exploration, we investigated the impact of image augmentation techniques on the dataset. To assess model performance, we built and compared four distinct prediction models: two convolutional neural networks (CNNs) built from scratch and two leveraging transfer learning. We then evaluated these models on their ability to accurately classify X-ray images. This involved analyzing which models achieved superior performance and identifying key hyperparameter settings that contributed to their success. We submitted the top three models to the COVID-19 Diagnostic AI Model Share competition to benchmark their performance against others. Following this initial submission, we continued our exploration, experimenting with additional models and comprehensively comparing their effectiveness. This in-depth analysis allowed us to definitively identify the best-performing models.

# Steps 
1. Data Exploration
2. Split the Dataset
3. Fine-tune VGG19, ResNet101 and DenseNet201
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



# Conclusion 



# References 


Chowdhury, M. E. H., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M. A., Mahbub, Z. B., Islam, K. R., Khan, M. S., Iqbal, A., Al-Emadi, N., Reaz, M. B. I., & Islam, T. I. (2020, June 15). Can ai help in screening viral and covid-19 pneumonia?. arXiv.org. https://arxiv.org/abs/2003.13145 

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2018, January 28). Densely connected Convolutional Networks. arXiv.org. https://arxiv.org/abs/1608.06993 

Kong, L., & Cheng, J. (2022, August). Classification and detection of COVID-19 X-ray images based on DenseNet and VGG16 feature fusion. Biomedical signal processing and control. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9080057/ 

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017, December 25). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv.org. https://arxiv.org/abs/1711.05225 

Wu, C.-J., Park , S., Cho, Y., Kim, Y., & et al. (2019, December). Short-term reproducibility of pulmonary nodule and mass detection in chest radiographs: Comparison among radiologists and four different computer-aided detections with convolutional neural net. Scientific reports. https://pubmed.ncbi.nlm.nih.gov/31822774/ 
