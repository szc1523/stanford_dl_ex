%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)    %??????????????????
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
m = size(x, 2);
z = W * x;
temp1 = sqrt((z).^2 + params.epsilon);
temp2 = W'*z - x;

cost = 1/m * (params.lambda * sum(sum(temp1)) + ...
  (1/2) * sum(sum(temp2 .^ 2)));

Wgrad = 1/m * (params.lambda * z./temp1*x' + W*temp2*x' + z*temp2');


% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);