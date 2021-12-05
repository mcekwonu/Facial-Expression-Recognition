# Facial-Expression-Recognition
Facial Expression Recognition trained with ResNet architecture in Pytorch

## sample of training dataset
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

Download dataset via: https://www.kaggle.com/msambare/fer2013

![train_sample](https://user-images.githubusercontent.com/17880412/144751501-aaf68689-bcdf-4801-ad3c-35aba6373f37.png)

## sample predictions with test dataset


![predicted_happy|100x100, 50%](https://user-images.githubusercontent.com/17880412/144752307-7c175dc5-c791-408b-87b8-fd9bd623632f.png)
![predicted_fear](https://user-images.githubusercontent.com/17880412/144752313-938a5099-a0b6-44c6-87a9-98585bdfb84a.png)
![predicted_surprise](https://user-images.githubusercontent.com/17880412/144752314-ca147c07-74a9-43e4-ad4d-a7b1d3e65126.png)
