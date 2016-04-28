function[J, grad] = costFunction(X, y, Theta1, Theta2, lambda)
% costFunctionReg returns the cost and gradient given the input 
% matrix X, output vector y, Theta1 and Theta2. This is for a
% neural network architecture with one hidden layer.

m = size(X, 1);
J = 0;

% Start forward propagation.
a1 = [ones(m, 1), X];
z2 = Theta1 * a1';
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Get the cost 
for i = 1:m
    h = a3(:,i);
    J = J + ((-y(i) * log(h)) - ((1-y(i)) * log(1 - h)));
end
Theta1_reg = Theta1(1:end, 2:end);
Theta2_reg = Theta2(1:end, 2:end);

J = J/m + ((lambda/(2 * m)) * sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

end