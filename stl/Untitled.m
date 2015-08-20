addpath(genpath('..'))

numClasses  = 5; % doing 5-class digit recognition
% initialize softmax weights randomly
randTheta2 = randn(numClasses, featureSize)*0.01;  % 1/sqrt(params.n);
randTheta2 = randTheta2 ./ repmat(sqrt(sum(randTheta2.^2,2)), 1, size(randTheta2,2)); 
randTheta2 = randTheta2';
randTheta2 = randTheta2(:);

%  Use minFunc and softmax_regression_vec from the previous exercise to 
%  train a multi-class classifier. 
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;

% optimize
%%% YOUR CODE HERE %%%
theta(:) = derivativeCheck(@softmaxcost, randTheta2, 1, 1, ...
  trainFeatures, trainLabels);


% theta=[theta, zeros(n,1)]; % expand theta to include the last class.
% [~,train_pred] = max(theta'*trainFeatures, [], 1);
% [~,pred] = max(theta'*testFeatures, [], 1);