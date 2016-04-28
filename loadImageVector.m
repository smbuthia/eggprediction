function[imageVector] = loadImageVector(imageFile, scale)
% LOADIMAGEVECTOR Scales an image file to a specified scale
% and returns the unrolled vector of the image matrix.
imageVector = [];
    % Read the image into a matrix.
    A = imread(imageFile);
    % Get the dimensions of the matrix - it is a 3D matrix if the image is RGB
    % and a 2D matrix if the image is grayscale or binary.
    if(size(size(A), 2) == 3) % Check if the image is in RGB
        % If the image is in RGB, it is first converted to grayscale.
        A = rgb2gray(A);
    endif
    if(size(size(A), 2) == 2) % Check if the image is in grayscale.
        % Resize the image to the given scale.
        B = imresize(A, scale);
        % Unroll the image matrix into a vector.
        imageVector = B(:);
    endif
end