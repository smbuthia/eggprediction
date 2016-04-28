function g = sigmoid(z)
% SIGMOID evaluates and returns the sigmoid value of any vector/matrix passed to it.
g = 1.0 ./ (1.0 + exp(-z));
end