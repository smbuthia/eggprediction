% This is the entry point file for the project

% Read the .png image files in the grayscale folder into a file name vector.
imageFilesList =  glob('grayscale/*.png');

% Instantiate the training set matrix X
X = [];
mAll = length(imageFilesList);
y = zeros(mAll,1);
% Randomize the file list.
randVec = randperm(mAll);
for i = 1:mAll
    fileName = imageFilesList(randVec(i)){1,1};
	% Replace the backslash character with forward slash to avoid errors due to special characters.
	fileName(fileName == "\\") = "/"; 
	% Load the image into a vector
	X(i,:) = loadImageVector(fileName, 0.01);
	
	% Check from the file name whether image is from male or female egg
	fileName = strsplit(fileName, '/');
	fname = fileName{1,2};
	% Female eggs are labelled 1 all others are 0 :)
	if(regexp(fname, 'F*') == 1)
	    y(i) == 1;
	else
	    y(i) == 0;
	endif
end

