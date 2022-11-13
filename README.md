# Targeted Data Augmentation for bias mitigation
In this article, we proposed a new and effective method for mitigating biases called Targeted Data Augmentation (TDA). Since removing biases is a tedious and difficult task, we proposed to insert them, instead.
First, we manually examined, identified, and annotated biases in two representative and diverse datasets - a dataset of clinical skin lesions and a dataset of male and female faces. 
Through Counterfactual Bias Insertion, we confirmed that the bias associated with the frame, ruler, and glasses strongly affects the models.
To immunize the models, we used Targeted Data Augmentation: in short, we modified the samples during training by randomly inserting biases. 
Our method resulted in a significant decrease in bias measures, more specifically, from a twofold to more than 50-fold improvement after training with TDA, with a negligible increase in the error rate.  

