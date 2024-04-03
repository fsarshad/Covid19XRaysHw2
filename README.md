# Covid-19 Chest X-Rays Deep Learning 
![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/894edaf6-47b7-4997-bfc1-e3dfa8650615)

# Content
1. [Introduction](#introduction)
2. [Overview](#Overview)
3. [Methods](#Methods) 
4. [Results](#Results)
5. [Conclusion](#Conclusion)
6. [References](#References)

# Introduction
The study below looks at utilizing artificial intelligence (AI) to rapidly and reliably detect COVID-19 in chest X-ray pictures. The authors created a huge dataset of chest X-rays, encompassing regular, viral pneumonia, and COVID-19 cases. They then used deep learning methods to build a system to evaluate X-rays and differentiate between these groups. 

Link: https://arxiv.org/abs/2003.13145

As a team of three, at Columbia University, our project will use deep learning to try and produce similar results. 

# Overview 

Our project aimed to develop accurate models for COVID-19 detection using chest X-rays. We began by analyzing sample images showcasing positive COVID-19 cases. Following data exploration, we investigated the impact of image augmentation techniques on the dataset. To assess model performance, we built and compared four distinct prediction models: two convolutional neural networks (CNNs) built from scratch and two leveraging transfer learning. We then evaluated these models on their ability to accurately classify X-ray images. This involved analyzing which models achieved superior performance and identifying key hyperparameter settings that contributed to their success. We submitted the top three models to the COVID-19 Diagnostic AI Model Share competition to benchmark their performance against others. Following this initial submission, we continued our exploration, experimenting with additional models and comprehensively comparing their effectiveness. This in-depth analysis allowed us to definitively identify the best-performing models.

# Methods  

Below, we created a pie chart to show the distribution of the X-Ray image categories. There are three image categories: COVID-19, Normal, and Viral Pneumonia. 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/e554b266-8d8e-4bb4-8739-f0c6db9e3198)

Normal Image showed a 67.26% sample size. COVID-19 Image showed a 23.86% sample size and Viral Pneumonia showed an 8.88% sample size. 

In the image below, you can see a random sample image for each category. On the left is COVID-19, in the middle is Normal, and on the right is Pneumonia. 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/049f3de0-b4de-48da-bc5c-591140220e60)

The COVID-19 image sample on the left, above, shows the chest is unhealthy compared to the Normal Image sample. The pneumonia image sample shows congestion in reference to the normal image. 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/452576a3-2cee-4674-bd13-28e15f9689e6)

To augment our dataset, we are artificially introducing sample diversity by applying random, yet realistic, transformations to the training images. The data augmentation techniques we apply to the training dataset, include random horizontal flipping, rotation, zooming, contrast adjustment, and translation.

By exposing the model to augmented examples during training, it learns to recognize objects/patterns regardless of their orientation, size, position, or lighting conditions within the input images. This enhanced invariance to transformations improves the model's ability to generalize to unseen data—preventing overfitting and encouraging learning more general features. Augmentations like adjustment to contrast make the model robust to noise/variations in real-world data. These preprocessing and augmentation techniques can significantly boost the model's performance, generalization capabilities, and robustness.

Using the given reference code as a starting point, we designed our CNN architecture with multiple convolutional blocks, each consisting of convolutional layers with different filter sizes and followed by max-pooling and dropout layers. The depth of the network increases gradually, with the number of filters doubling at each block (32, 64, 128, 256...). This allows the network to learn increasingly complex and abstract representations as it progresses through the layers. The use of smaller 3x3 filters after the initial 5x5 filters in each block is a common pattern that helps the network learn more complex features in a hierarchical manner. The final layers are fully connected dense layers, which combine the high-level features learned by the convolutional blocks for classification.

For the first custom model, we did not include any explicit regularization techniques like L2 regularization or dropout layers. This was intentional to observe the model's performance without any regularization and to establish a baseline for comparison. As mentioned earlier, the model exhibited signs of overfitting, which we aim to address in the subsequent models (See Custom Models 2 and 3).

Initially, for the first iteration of our custom model, we used the default values for the learning rate (0.001) and batch size (32). However, as discussed above, we observed something peculiar in the loss curves: the loss was increasing over time, which indicated that the model had trouble converging to a local/global minima. As a result, we decided to experiment with a smaller learning rate (0.0001) to see if it would help the model converge better.

For the loss function, we chose categorical cross-entropy, which is a standard choice for multi-class classification tasks. This loss function measures the performance of the model by comparing the predicted probabilities for each class with the true class labels, and it is well-suited for problems where the classes are mutually exclusive.

For the optimizer, we used Adam, which is a popular choice for its adaptive learning rate and good convergence properties. At a high level, Adam combines the benefits of momentum and RMSProp, adapting the learning rate for each parameter based on the gradients and their moments. This often leads to faster convergence and better performance compared to other optimizers like stochastic gradient descent (SGD).

Custom Model 2 

In contrast to custom model 1, this one incorporates several more sophisticated techniques and design choices. Instead of employing standard ReLU activation functions, here, we utilize the GELU (Gaussian Error Linear Unit) activation function for all convolutional and dense layers, except for the final output layer which uses softmax. GELU is a smoother, more continuous shape than ReLU and provides better performance as seen in OpenAI's GPT models.

One challenge we ran into was an issue with exploding gradients, which caused the loss value to randomly spike from ~1 to double or even triple digits. To mitigate this, we employ the He Normal (HeNormal) kernel initializer for all convolutional and dense layers, which can help stabilize the training process by initializing the weights in a way that prevents exploding gradients. Additionally, we use gradient clipping on our Adam optimizer with a norm value of 1.0 to prevent the gradients from becoming too large during training, which can lead to unstable training and convergence issues.

Whereas model 1 did not consist of any explicit regularization techniques, in the second implementation of our model, we implemented several regularization techniques and callbacks.

L2 regularization (weight decay) was applied to the kernel weights of all convolutional and dense layers. This adds a penalty term to the loss function that encourages smaller weight values, which can help prevent overfitting.

Dropout layers were also used after each convolutional block and before the final dense layers. As discussed in class, dropout randomly sets a fraction of input units to zero during training, effectively breaking the co-adaptation of neurons and forcing the network to learn more robust and redundant representations.

The EarlyStopping callback was used to monitor the validation loss and stop training if it didn't improve for a certain number of epochs. We used a patience of 10 steps (i.e. if the validation loss didn't improve for 10 steps, we stopped training). This prevents the model from overfitting to the training data once it has converged.

The ReduceLROnPlateau callback was also implemented to monitor the validation loss and reduce the learning rate by a factor of 0.1 if the validation loss didn't improve for a certain number of epochs (patience=5 in this case). This can help the model escape local minima and potentially achieve better performance.

For the second custom model, we didn't alter too many hyperparameters except for the batch size—reducing that from 32 to 16. This was done to see if a smaller batch size would help the model generalize better and prevent overfitting. We also kept the learning rate at 0.0001, as it seemed to work well in the first model. The other hyperparameters were kept the same as the first model to maintain consistency and isolate the effects of the regularization techniques.

Custom Model 3 

Building on models 1 and 2, model 3 utilizes the Swish activation function instead of ReLU or GELU, which has been shown to perform well in various deep-learning tasks, particularly in computer vision applications. The LeCun Normal kernel initializer is employed for all convolutional layers, as it is specifically designed for non-linear activation functions like Swish. For the dense layers, the Xavier Normal initializer is used, which helps maintain the scale of the gradients during backpropagation. We referenced Hanin et al., 2018 for these design choices around kernel initializers.

Notably, in this model, we also incorporate batch normalization layers after each convolutional layer and before the activation function. Batch normalization can aid in training convergence by reducing internal covariate shifts and can also act as a regularizer, potentially improving the model's performance.

# Results 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/65d60af7-74ff-4062-a626-68c52e2ac32c)

At first glance, it's clear that after epoch 8, the training loss is consistently lower than the validation loss, and in fact, the validation loss is actually increasing. This suggests that the **model is overfitting to the training data**. To address this, in the second custom model, we will implement a custom CNN model with dropout layers and regularization techniques to prevent overfitting (more on this later).

Furthermore, while the rest of the plots look reasonable enough, **initially**, an interesting observation we made was that **both our training and validation loss curves counterintuitively increased over time** (as epoch number increases). This puzzled us, but we ultimately realized that this was due to the fact that the **Adam optimizer's default learning rate of 1e-3 was too high** for our model architecture and the task at hand. To address this and have some decently satisfactory results for our first model, we decided to reduce the learning rate to 1e-4. This helped us achieve a more stable training process and better convergence, though much more tuning was needed to improve the model's performance—as we'll see in future models. 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/c73ef7ef-06a2-4d69-be41-d8f3d6b79903)

Interestingly enough, despite being more complex than the first custom model, the second custom model performed worse in terms of validation accuracy (~78% for model 1 vs. ~68% for model 2). This was a bit surprising to us, but it was a good learning experience to see that more complexity does not always equate to better performance. In this case, the model likely suffered from overfitting due to the increased number of parameters and layers, which could not be effectively regularized by the dropout layers alone. 

Furthermore, another puzzling finding is that after the 8th epoch, the training loss seemingly randomly spikes upward, which is likely an indication of a potential exploding gradient problem. This could be due to the increased complexity of the model, which requires more effective mitigation than just weight initialization and dropout layers. 

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/ee9a0077-9729-44dd-a898-762f228fa635)

According to our loss and accuracy curves, it seems that model 3 is the best-performing of our 3 custom models so far, approaching ~85% validation accuracy. This makes sense given the fact that this is the most complex model we've built yet, with a deeper architecture (12 Conv2D layers), more regularization techniques, and a more sophisticated design involving complex callbacks. 

However, we do notice that overfitting is still a concern, as the training loss is consistently lower than the validation loss, and the training accuracy is higher than the validation accuracy—with the latter acting a bit erratic. This suggests that the model is likely learning the training data too well and not generalizing to unseen data. To address this, we may need to further increase the dropout rate, add more regularization, or perhaps improve our data augmentation pipeline.

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/40349504-873d-4029-b877-849b2e9a248a)

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/048e537f-14b2-418a-9fcd-44caddfaf411)

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/c881ff56-5ea6-49f2-b23c-330054b63480)

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/f9b971c3-3b60-4b9e-a140-480dc26d6eb9)

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/58207364-ba7a-4668-a277-63be4084add9)

![image](https://github.com/fsarshad/Covid19XRaysHw2/assets/51839755/9515c4ed-5fc3-4961-b071-fc70a043e43e)

# Conclusion 

In all honesty, the biggest challenge we faced was managing our developer environment(s). Since we were instructed to initially use Google Colab, we mainly developed on that platform. However, we encountered issues with the runtime limits which made it difficult to train large models without Colab giving us Runtime disconnected errors "due to inactivity or reaching its maximum duration." Since we also did not have Colab Pro or access to Google Cloud credits via the course, we had to switch over to running everything locally on our machines which was a bit of a hassle—especially since we all have different machines (2 Macs and 1 Windows). This required us to install all the necessary libraries and dependencies on our local machines, which took some time to get right. We also had to deal with issues related to GPU memory and runtime limits, which required us to optimize our code and data pipeline to make the most of the available resources. Overall, this experience taught us the importance of adaptability and problem-solving skills when dealing with technical challenges, as well as the need for effective resource management in machine learning projects.

# References 

Chowdhury, M. E. H., Rahman, T., Khandakar, A., Mazhar, R., Kadir, M. A., Mahbub, Z. B., Islam, K. R., Khan, M. S., Iqbal, A., Al-Emadi, N., Reaz, M. B. I., & Islam, T. I. (2020, June 15). Can ai help in screening viral and covid-19 pneumonia?. arXiv.org. https://arxiv.org/abs/2003.13145 

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2018, January 28). Densely connected Convolutional Networks. arXiv.org. https://arxiv.org/abs/1608.06993 

Kong, L., & Cheng, J. (2022, August). Classification and detection of COVID-19 X-ray images based on DenseNet and VGG16 feature fusion. Biomedical signal processing and control. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9080057/ 

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017, December 25). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv.org. https://arxiv.org/abs/1711.05225 

Wu, C.-J., Park , S., Cho, Y., Kim, Y., & et al. (2019, December). Short-term reproducibility of pulmonary nodule and mass detection in chest radiographs: Comparison among radiologists and four different computer-aided detections with convolutional neural net. Scientific reports. https://pubmed.ncbi.nlm.nih.gov/31822774/ 
