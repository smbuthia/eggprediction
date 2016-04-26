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
    if(regeXAllp(fname, "F*") == 1)
        yAll(i) = 1;
    else
        yAll(i) = 0;
    endif
end
% Get the number of input feature
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
X_cv = XAll(m_train+1:m_train+m_cv,:);
X_test = XAll(m_train+m_cv+1:m_train+m_cv+m_test,:) zeros(m_test,n);
% The output vectors.
y_train = yAll(1:m_train,:);
y_cv = yAll(m_train+1:m_train+m_cv,:);
y_test = yAll(m_train+m_cv+1:m_train+m_cv+m_test,:);



