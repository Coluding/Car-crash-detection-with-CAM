# insurance_image_recog

## 1. Goal
The goal of this repo is to classify car crash images into crash and non crash. Additionally, it should be shown where the crash is found on the image. This is done via Class Activation Maps (CAMs) without gradients

## 2. Structure
### 2.1 Training of the model
The model is trained using PyTorch and a config file. With the config file, new models can be built very easily using a setup like this:

```yaml
resnet:
    epochs: 100
    
    batch_size: 16
    learning_rate: 0.001
    train_backbone_params: True
    train_test_split_ratio: 0.8
    activation_function: 'tanh'
    classifier_layer:
        - classifier_layer_1:
            - 512
            - 256
            - 0.2 # Dropout rate
        - classifier_layer_2:
            - 256
            - 6
            - 0
 ```
 - The model can be trained in the Azure Cloud if the correct settings are made and the Azure connection is established. 
 - The training process is monitored via MLFlow.
 - If possible GPUs will be used.
 - The final model class can be used to export the trained model. It contains a method that computes the CAM of the feature map of the last convolutional layer.
 - The model has an validation accuracy of 100%.
 
 ### 2.2 Deplyoing the model
 The fully trained model is deployed on some local endpoint. This will be improved in the future.
 
 
