function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m,1) X]; %adding bias
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
h = sigmoid(z3);

yh = zeros(m,num_labels); %y hot encoded
for i=1:m
    yh(i, y(i)) = 1;
end

 J = -yh .* log(h) - (1-yh).*log(1-h);
 J = sum(J,1);
 J = 1/m * sum(J,2);
 fprintf("%d",J(1,1));
 
 theta1_without_b = Theta1(:, 2:end);
 theta2_without_b = Theta2(:, 2:end);
 regularization= sum( sum(theta1_without_b .^2,1) ,2) + sum( sum( theta2_without_b .^2,1) ,2);
 regularization = (lambda/(2*m)) * regularization;
 
 J = J + regularization;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% Theta1_grad = zeros(hidden_layer_size,input_layer_size);
% Theta2_grad = zeros(num_labels,hidden_layer_size + 1);

% for i=1:2
%     Xi = [1; X(i, :)']; %adding bias (401,1)
%     %size(Xi)
%     z2 = Theta1 * Xi; % (25,401) * (401,1) = (25,1)
%     a2 = sigmoid(z2); % (25,1)
%     
%     a2 = [1; a2]; % (26,1)
%     z3 = Theta2 * a2; % (10,26) * (26,1) = (10,1)
%     a3 = sigmoid(z3); % (10,1)
%         
%       yy = 1:num_labels == y(i);
%     
%     delta3 = a3-yy'; %(10,1)
%     
%     delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)]; % (26,10) * (10,1) = (26,1)
%     delta2 = delta2(2:end);
%     
% %     size(delta2)
% %     size(Xi(2:end, :)')
%     Theta1_grad = Theta1_grad + delta2 * Xi' ; %(25,1) * (1,400) = (25,400)
%     Theta2_grad = Theta2_grad + delta3 * a2' ; % (10,1) * (1,26) = (10, 26)
% end


for t = 1:m
	a1 = [1; X(t,:)'];

	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yy = ([1:num_labels]==y(t))';
	% For the delta values:
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end);

	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end


% 
% Theta1_grad = Theta1_grad .* 1/m;
% Theta2_grad = Theta2_grad .* 1/m;

   
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
