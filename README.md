# HappyEarth

## Inspiration
There has been a lot of waste generation over the past decade, and this is leading to global warming. Humans have, without a doubt, contributed the maximum towards global warming. However, some people have no clue that they're contributing to this global issue. Thus, hopefully, to help people make better, wiser and empathetic decisions, we came up with this app idea.

## What it does
It takes a single image as input, runs it through the machine learning model at the backend, and gives information about the object, such as name, whether it is reusable, carbon footprint, etc.

## How we built it
- Haramrit began working on the machine learning model, while Simran was responsible for collecting the data and writeups. 
- First, we had to collect images and label them appropriately. For this, we found an already existing dataset on Kaggle, and we just modified its classes. We created multiple directories, one for each object and put train and test images in their respective directories. We used ~150 images for each object for the training process.
- For the model, we used DenseNet which is a well-known pre-trained CNN model. For training the model, we used Kaggle, since it allows GPU training, and images are trained faster using GPUs than using CPUs. (Our best model got a test accuracy of ~83%, however, the numbers are not very accurate as the dataset was quite small.)
- After training and saving the trained model, we used Streamlit to create a simple web app which serves as a GUI to upload an image which is then fed to the ML model. This model, if it is able to predict a given object with >70% confidence, it gives the desired output. If, however, it is not able to detect the image properly, it lets the user know that it was not able to detect the object in the uploaded image and fails gracefully.
- Then we configured our app for Heroku and deployed it on to the cloud.

## Challenges we ran into
- Had trouble figuring out installing PyTorch on Heroku
- Had to manually create the labels and put images in their respective directories - was very time consuming

## Accomplishments that we're proud of
- Being able to finish a project (almost 70-80%) which was completely outside our domain
- Got to work on an image classification project for the first time
- Hopefully our app plays a role, however small, to contribute to the well-being of this planet

## What we learned
- How to implement a simple image classifier using PyTorch
- How to sync well as a team

## What's next for Don't Trash Me
- Can work on expanding dataset, adding new objects, dataset is very small at the moment
- Add functionality to detect objects from camera live stream
- Add multiple-image upload functionality
- More useful ways of reusing existing objects
