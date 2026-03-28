# Egg prediction prototype

## A prototype that uses image processing and neural networks algorithms to predict the sex of a chick from the egg shape

The prototype takes a sample set of egg photos labelled male and female to use as a training set. 

These photos are then processed into gray-scale images from which 2 dimensional matrices are generated. This makes processing easier and more efficient as what is relevant is the shape of the egg.

The matrices and corresponding labels are applied to a neural network architecture from which a set of weights is generated that will be used to predict the expected sex from the image of an egg.

## Archive Note

This project was inspired by research in Ethiopia and Kenya on farmers who said they were able to identify the sex of chicks before they hatched based on the shape of the egg. Not much had been published at the time, but it felt like a good use case for machine learning. 

Unfortunately, we never got a big enough data set to prove or disprove this hypothesis. Taking photos of the eggs meant excessive handling that often resulted in them not hatching, meaning most hatcheries or farmers were not comfortable with that.

