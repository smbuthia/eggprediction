% This is the entry point file for the project
% Clear all variables.
clear;
close all;
clc;
% Read the .png image files in the grayscale folder into a file name vector.
imageFilesList =  glob('grayscale/*.png');

% Instantiate the training set matrix X
XAll = [];
mAll = length(imageFilesList);
yAll = zeros(mAll,1);
% Randomize the file list.
randVec = randperm(mAll);
for i = 1:mAll
    fileName = imageFilesList(randVec(i)){1,1};
    % Replace the backslash character with forward slash to avoid errors due to special characters.
    fileName(fileName == "\\") = "/";
    % Load the image into a vector
    XAll(i,:) = loadImageVector(fileName, 0.01);
    
    % Check from the file name whether image is from male or female egg
    fileName = strsplit(fileName, '/');
    fname = fileName{1,2};
    % Female eggs are labelled 1 all others are 0 :)
    if(regexp(fname, "F*") == 1)
        yAll(i) = 1;
    else
        yAll(i) = 0;
    endif
end
% Get the number of input features
n = size(XAll)(:,2);

% We now have the input matrix XAll and output vector yAll.
% We need to split these into a training set (X_train, y_train), 
% cross-validation set (X_cv, y_cv), and test set (X_test, y_test).
% The ratios are 60:20:20 respectively.
m_train = 0.6 * mAll;
m_cv = 0.2 * mAll;
m_test = m_cv;
% The input matrices.
X_train = XAll(1:m_train,:);
X_cv = XAll((m_train+1):(m_train+m_cv),:);
X_test = XAll((m_train+m_cv+1):(m_train+m_cv+m_test),:);
% The output vectors.
y_train = yAll(1:m_train,:);
y_cv = yAll((m_train+1):(m_train+m_cv),:);
y_test = yAll((m_train+m_cv+1):(m_train+m_cv+m_test),:);

fprintf(['All training data, cross-validation data,'... 
        'and test data is loaded \n'...
		'\nPress any key to proceed\n']);
pause;

% Some parameters that we will use
input_layer_size = n;
hidden_layer_size = 30;
final_layer_size = 1; % the output will be a 0 or a 1




% We now initialize Theta1 and Theta2.
% The architecture of the neural network is such that it has three layers.
% One input layer, one hidden layer, and one output layer with one unit.
init_theta1 = initializeWeights(input_layer_size, hidden_layer_size);
init_theta2 = initializeWeights(hidden_layer_size, final_layer_size);

init_weights = [init_theta1(:); init_theta2(:)];

fprintf('Random weights initialized\n\nPress any key to continue.\n');
pause;

% Regularization parameter 
lambda = 0;

% The calculated cost
J = costFunctionReg(init_weights,...
                    input_layer_size,...
					hidden_layer_size,...
					final_layer_size,...
					X_train, y_train, lambda);

fprintf(['\nInitial cost is found to be: %f'...
        '\n\nPress any key to continue\n'], J);
pause;

% The cost function to be minimized becomes
costFunction = @(input_params) costFunctionReg(input_params,...
                                         input_layer_size,...
										 hidden_layer_size,...
										 final_layer_size,...
										 X_train, y_train, lambda);
										 
options = optimset('MaxIter', 50);
										 
[params, cost] = fmincg(costFunction, init_weights, options);

% From the params, reshape to obtain our final weights
Theta1 = reshape(params(1:(hidden_layer_size * (input_layer_size + 1))),...
                 hidden_layer_size,...
				 (input_layer_size + 1));

Theta2 = reshape(params(((hidden_layer_size * (input_layer_size + 1)) + 1):end),...
                 final_layer_size,...
				 (hidden_layer_size + 1));
				 
% Predict the outcome against the training set
prediction = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(prediction == y_train)) * 100);