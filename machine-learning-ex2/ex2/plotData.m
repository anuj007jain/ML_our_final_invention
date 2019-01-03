function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


X1 = X(:,1);
X2 = X(:,2);
X3 = X1;
X4 = X2;
X1(X1.*y == 0) = [];
X2(X2.*y == 0) = [];
plot(X1,X2,'+');
hold on;
X3(X3.*y ~= 0) = [];
X4(X4.*y ~= 0) = [];
plot(X3,X4,'o');





% =========================================================================



hold off;

end
