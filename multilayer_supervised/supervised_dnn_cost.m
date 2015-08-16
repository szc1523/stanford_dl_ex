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
gradStack = cell(numHidden+1, 1); %grad of each layer
delta = cell(numHidden+1, 1); %delta of each layer

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
        pred_prob = hAct{l}; %its 10 by m matrix, the output prediction
    end
end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
% warning: there has to be a (1/m) in the cost function
% because there is (1/m) in the gradient function,
% two has to be same. both have or neither have.
% the same with (1/m) before reg terms.
I = sub2ind(size(hAct{numHidden+1}), labels', 1:m); 
cost = -(1/m)*sum(log(hAct{numHidden+1}(I)));  %softmax cost
for l = 1 : numHidden+1
    cost = cost + ei.lambda/2*sum(sum((stack{l}.W).^2)); % does regularization need 1/m?????
end

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% compute delta and grad of output layer
ym = zeros(ei.output_dim,m);  %ym is y matrix
ym(I) = 1;
delta{numHidden+1} = hAct{numHidden+1}-ym;
gradStack{numHidden+1}.b = (1/m)*sum(delta{numHidden+1},2);
gradStack{numHidden+1}.W = (1/m)*delta{numHidden+1}*(hAct{numHidden})'...
                            +ei.lambda.*stack{numHidden+1}.W;
% compute delta and grad of hidden layer
for l = numHidden : -1 : 1
    %calculate delta
    %delta only backprop to real neuron, not to bias neuron
    delta{l} = ((stack{l+1}.W)'*delta{l+1})...  %UFLDL seems wrong
        .*(hAct{l}.*(1-hAct{l}));    
    %calculate gradient
    gradStack{l}.b = (1/m)*sum(delta{l},2);
    if l ~= 1        
        gradStack{l}.W = (1/m)*delta{l}*(hAct{l-1})'+ei.lambda.*stack{l}.W;
    else
        gradStack{l}.W = (1/m)*delta{l}*data'+ei.lambda.*stack{l}.W;
    end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
% its in the cost term

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



