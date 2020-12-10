# Semantic-Segmentation-of-Urban-Scene-Images-Using-Recurrent-Neural-Networks
# Abstract [en]
# Background: In Autonomous Driving Vehicles, the vehicle receives pixel-wise sensor data from RGB cameras, point-wise depth information from the cameras, and sensors data as input. The computer present inside the Autonomous Driving vehicle processes the input data and provides the desired output, such as steering angle, torque, and brake. To make an accurate decision by the vehicle, the computer inside the vehicle should be completely aware of its surroundings and understand each pixel in the driving scene. Semantic Segmentation is the task of assigning a class label (Such as Car, Road, Pedestrian, or Sky) to each pixel in the given image. So, a better performing Semantic Segmentation algorithm will contribute to the advancement of the Autonomous Driving field.

# Research Gap: Traditional methods, such as handcrafted features and feature extraction methods, were mainly used to solve Semantic Segmentation. Since the rise of deep learning, most of the works are using deep learning to dealing with Semantic Segmentation. The most commonly used neural network architecture to deal with Semantic Segmentation was the Convolutional Neural Network (CNN). Even though some works made use of Recurrent Neural Network (RNN), the effect of RNN in dealing with Semantic Segmentation was not yet thoroughly studied. Our study addresses this research gap.

# Idea: After going through the existing literature, we came up with the idea of “Using RNNs as an add-on module, to augment the skip-connections in Semantic Segmentation Networks through residual connections.”

# Objectives and Method: The main objective of our work is to improve the Semantic Segmentation network’s performance by using RNNs. The Experiment was chosen as a methodology to conduct our study. In our work, We proposed three novel architectures called UR-Net, UAR-Net, and DLR-Net by implementing our idea to the existing networks U-Net, Attention U-Net, and DeepLabV3+ respectively.

# Results and Findings: We empirically showed that our proposed architectures have shown improvement in efficiently segmenting the edges and boundaries. Through our study, we found that there is a trade-off between using RNNs and Inference time of the model. Suppose we use RNNs to improve the performance of Semantic Segmentation Networks. In that case, we need to trade off some extra seconds during the inference of the model.

# Conclusion: Our findings will not contribute to the Autonomous driving field, where we need better performance in real-time. But, our findings will contribute to the advancement of Bio-medical Image segmentation, where doctors can trade-off those extra seconds during inference for better performance.

# Publication Link: https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1494982&dswid=-5063
Instructions for replicating the experiment:
>> Choose any operating system of your choice and make sure you installed all the below mentioned pacakages.
   1. Programming Language : Python 3.6
   2. Deep Learning Framework : PyTorch 1.3.1 3. Environment : Jupyter Notebook
   4. Cuda Version : 11.0
   5. preferred installer program (PIP) Libraries :
        (a) Albumentations 0.4.5 
        (b) H5py 2.8.0
        (c) Matplotlib 3.1.3 
        (d) Numpy 1.18.1
        (e) Pandas 1.0.2 
        (f) Pillow 6.2.0
        (g) Pip 20.0.2
        (h) Scikit-Image 0.16.2
        (i) Scikit-learn 0.22.1
        (j) Torch Summary 1.5.1 
        (k) Torchvision 0.4.2
        
>> Download the Cityscapes dataset and the experiment source code that you want to repilicate.
>> Make sure that source code and the Cityscapes dataset are in the same directory. 
>> Here you can run the code to train the model and get results or you can also donload the pre-trained weights from here (https://studentbth-my.sharepoint.com/:f:/g/personal/veda18_student_bth_se/Et-Qnxr3wWpOsIQqSayieCMBxek0h-uj6O7lMuvmsQl61g?e=PJbjTy)
>> If you choose to use pre-trained weights, then make sure that pre-trained weights directory in the source code.
>> If you faced any problems during repilication of the experiment, then you can create an issue here and we will try to resolve it as soon as possible. 
