function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%batch
for iter=1:num_iters
theta=theta-alpha/m*X'*(X*theta-y);
J_history(iter)=computeCostMulti(X,y,theta);
end

%stochastic
% for iter=1:num_iters
% h=X*theta-y;
% for j=1:m
% theta=theta-alpha*(h(j,1)*X(j,:))'; %注意这里不用×1/m,因为stochastic不是相加求和
% end
% J_history(iter)=computeCostMulti(X,y,theta);
% end



