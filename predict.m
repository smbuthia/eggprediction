function p = predict(Theta1, Theta2, X)

m = size(X, 1);
output_size = size(Theta2, 1);

% Instantiate predicted output
p = zeros(m, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dud, p] = max(h2, [], 2);


end