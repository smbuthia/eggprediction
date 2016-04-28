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

% TODO: Double check this portion.
% Initialize the big delta values.
cap_delta_1 = zeros(size(Theta1));
cap_delta_2 = zeros(size(Theta2));
for t = 1:m
    a_1 = [1, X(t,:)];
    z_2 = Theta1 * a_1';
	a_2 = [1; sigmoid(z_2)];
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);
    delta3 = a_3 - y_new(:,t);
	delta2 = Theta2' * delta3 .* sigmoidGradient([1; z_2]);
	delta2 = delta2(2:end);
	cap_delta_1 = cap_delta_1 + (delta2 * a_1);
	cap_delta_2 = cap_delta_2 + (delta3 * a_2');
end

Theta1_grad = cap_delta_1/m;
Theta1_grad_reg = Theta1_grad(1:end, 2:end) + (lambda/m * Theta1(1:end, 2:end));
Theta1_grad = [Theta1_grad(1:end, 1), Theta1_grad_reg];
Theta2_grad = cap_delta_2/m;
Theta2_grad_reg = Theta2_grad(1:end, 2:end) + (lambda/m * Theta2(1:end, 2:end));
Theta2_grad = [Theta2_grad(1:end, 1), Theta2_grad_reg];

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end