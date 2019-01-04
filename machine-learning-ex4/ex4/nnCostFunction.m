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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1 = 25*401
% Theta2 = 10*26
X_copy = X;
X_copy = [ones(size(X,1),1),X_copy];
A = sigmoid(Theta1*X_copy'); % 25*5000
A = [ones(1,size(X,1));A]; % 26*5000
H = sigmoid(Theta2*A); % 10*5000
y_new = zeros(num_labels,size(y,1)); % 10*5000
for i=1:size(y,1),
  y_new(y(i),i) = 1;
endfor
J = (-1/m) * sum(sum(y_new.*log(H) + (1-y_new).*log(1-H))) + (lambda/(2*m))*((sum(sum([Theta1(:,2:size(Theta1,2))].^2))) + sum(sum([Theta2(:,2:size(Theta2,2))].^2)));

Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));
for t=1:m,
  y_new = zeros(num_labels,1);
  y_new(y(t)) = 1;
  Ai1 = X(t,:); % 1*400
  Ai1 = [1, Ai1]; % 1*401
  Zi2 = Theta1*Ai1'; %25*1
  Ai2 = sigmoid(Zi2); %25*1
  Ai2 = [1 ; Ai2]; % 26*1
  Zi3 = Theta2*Ai2; % 10*1
  Ai3 = sigmoid(Zi3); % 10*1
  delta3 = Ai3 - y_new; % 10*1
  delta2 = (Theta2'*delta3).*sigmoidGradient([1;Zi2]); %25*1
  delta2 = delta2(2:size(Theta2,2),:);
  Delta2 = Delta2 + delta3*Ai2'; % 10*26
  Delta1 = Delta1 + delta2*Ai1; % 25*401
endfor

Theta2_grad = (1/m)*Delta2; % 10*26
Theta1_grad = (1/m)*Delta1; % 25*401

Theta2_grad = [Theta2_grad(:,1),Theta2_grad(:,2:size(Theta2_grad,2)) + (lambda/m).*Theta2(:,2:size(Theta2,2))];
Theta1_grad = [Theta1_grad(:,1),Theta1_grad(:,2:size(Theta1_grad,2)) + (lambda/m).*Theta1(:,2:size(Theta1,2))];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
