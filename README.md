# Targeted Data Augmentation for bias mitigation

🔗 Paper: [Targeted Data Augmentation for bias mitigation, Agnieszka Mikołajczyk-Bareła, Maria Ferlin, Michał Grochowski](https://arxiv.org/abs/2308.11386)

## Introduction
Data augmentation (DA) plays a vital role in enhancing the efficiency of deep learning-based systems. However, the existence of bias can distort or undermine the results of machine learning models without being immediately apparent. Removing bias from data is a challenging and time-consuming task, but it is essential to ensure the validity and fairness of machine learning models. Unfortunately, the traditional approach of removing bias can lead to new artifacts or residual biases, which are difficult to address even with advanced techniques like image inpainting.

That's why we propose a novel and revolutionary approach to deal with biases in machine learning models. Rather than eliminating biases, our proposed method, called Targeted Data Augmentation (TDA), enriches training sets with selected biases to force the model to learn to ignore them. By randomly adding biases to the input during training, the model learns to distinguish between relevant and irrelevant features. This approach is groundbreaking, as it breaks the cycle of mistaking correlation with causation by disrupting spurious correlations.

## Methodology
The proposed methodology behind TDA consists of four steps: bias identification, augmentation policy design, training with data augmentation, and model evaluation. We utilized a supervised step to detect any possible unwanted biases, and we used manual data exploration to achieve this. We manually labeled 2000 skin lesion images and automatically annotated the entire gender dataset with a trained glasses detection model. Based on the detected biases, we proposed an augmentation policy that mimicked the biases and injected them into the training data.

## Results
After training the model with this augmented data, we measured the bias using the Counterfactual Bias Insertion (CBI) method, which is a state-of-the-art method for bias evaluation. Our method showed a significant reduction in bias measures, with two to over fifty times fewer images switching classes after training with TDA, without significantly increasing the error rate.

## Contribution
Our contribution is three-fold: first, we propose a novel bias mitigation method that can easily complement the machine learning pipeline. Second, we present a bias mitigation benchmark that includes two publicly available datasets (over 2000 skin lesion and 50k gender images annotated by us), the code for TDA and bias evaluation, detailed results, and prepared collections of masks and images for bias testing. Third, we identify and confirm previously unknown biases in the gender classification dataset, analyze the robustness of popular models against bias, and show that standard evaluation metrics are often insufficient for detecting systematic errors in data.

## Conclusion
In conclusion, our research introduces a groundbreaking approach to deal with biases in machine learning models. Our proposed TDA method is state-of-the-art, and the results show that it is an effective way to reduce biases without increasing the error rate. Our contribution includes a bias mitigation benchmark that provides a useful resource for researchers and practitioners to detect and mitigate biases in their models. Overall, this paper demonstrates that the traditional approach to removing bias from data is not always the best way, and we encourage researchers to explore new methods like TDA to improve the validity and fairness of their machine learning models.
