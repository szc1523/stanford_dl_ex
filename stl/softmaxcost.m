function [f, g, preds] = softmaxcost( theta, X, y, pred)
% softmax This function is the costfunction of softmax. its different from
% the one in ex1 folder, in that it has full num_classes instead of
% num_classes - 1. Although the latter one saves computation time. it
% results some nasty bugs in the whole program ,eg. minFun. 
 
% Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-num_classes matrix.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.

if ~exist('pred','var')
  pred = false;
end;
m=size(X,2);
n=size(X,1);

% theta is a vector;  need to reshape to n x num_classes.
theta=reshape(theta, n, []);
num_classes=size(theta,2);

%% forward propagation
h = exp(theta'*X);  % no bias term
probs = bsxfun(@rdivide, h, sum(h));

%calculate cost
I = sub2ind(size(probs), y, 1:m); 
f = (-1/m)*sum(log(probs(I)));  

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    g = 0;
    return;
end;

%% backpropagation
ym = zeros(num_classes,m);  %ym is y matrix
ym(I) = 1;

g = (-1/m)*X*(ym-probs)';
g = g(:);

end

