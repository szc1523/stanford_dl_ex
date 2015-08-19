function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%

%% Step 0: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.
%%% YOUR CODE HERE %%%
m = size(x, 2);       
avg = mean(x, 2);     %ufldl seems to be wrong
x = bsxfun(@minus, x, avg);

%% Step 1: Apply svd
sigma = x * x' ./ m;
[U,S,V] = svd(sigma);

%% Step 1: Apply zca
epsilon = 1e-2; 
xZCAWhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
Z = bsxfun(@plus, xZCAWhite, avg); 
