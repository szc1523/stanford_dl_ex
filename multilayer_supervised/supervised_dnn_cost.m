function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%output pred_prob is prediction probability

%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false; %means prediction only
if exist('pred_only','var')
  po = pred_only;   %if pred_only is true, only feedforward
end;
%some constant
m = size(data,2);

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1); %activation of hidden layer
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%
for l = 1 : numHidden+1
    if l > 1
        z = [stack{l}.b,stack{l}.W]*[ones(1,m);hAct{l-1}];
    else
        z = [stack{l}.b,stack{l}.W]*[ones(1,m);data];
    end
    if l ~= numHidden+1
        hAct{l} = 1./(1+exp(-z)); %logistic
    else 
        h=exp(z);
        hAct{l} = bsxfun(@rdivide, h, sum(h)); %softmax
    end
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  pred_prob = hAct{numHidden+1};  %its 10 by m matrix
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



