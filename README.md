# FT_LINEAR_REGRESSION
The aim of this project is to introduce you to the basic concept behind machine learning. For this project, you will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm. 


## Project description

First of the projects in the machine learning area at 42 network. Objective is to create a linear regression function in the language of choise **(python3)**, train the model on the given dataset, save generated indexes and use them to predict car price depending on it's mileage. 

`train.py` functions with `csv` files with `,` as separator
## Plot Data after Training
![Screenshots](/pic/LR-Graph.png)
## Live Plotting of Training
![Screenshots](/pic/LR-Live.gif)
## Live Progress of Training from terminal
![Screenshots](/picFT_LINEAR_REGRESSION_TRAINING)
## Scatter predicted price
![Screenshots](/pic/PredictGraph.png)

## [Subject](SUBJECT.ft_linear_regression.en.pdf)


## [Linear Regression - Wiki](https://en.wikipedia.org/wiki/Linear_regression)

## Usage

Clone and change directory to project, then
	
	python3 train.py [flags]
	python3 predict.py [your_desired_mileage [path/to/thetas.txt] [flags2]]

train.py : If no file is passed as a parameter, function takes 'data.csv' as default file
### Flags

	-po 	- plot standardized dataset and solution
	-pn 	- plot original dataset and solution
	-hs 	- plot history of COST over iterations
	-l      - set learning rate (affects speed of learning), must be followed by a number
    -it 	- set number of iterations (affects accuracy of result), must be followed by a number, by default its uncapped
    -in     - --input : takes the name of datasets file 
    -o      - --output: the output file name of the THETAS
    -lv     - --live : take snapshots on every loop during the iteration to form a GIF showing the progress of the training live
      
### Flags2

	-sc 	- scatter the predicted price with datasets values, must be followed with path o datasets file
