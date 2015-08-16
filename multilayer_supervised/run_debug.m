%Sunzhicheng
%For debug dnn
% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
clear;clc; close all;
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [15, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% call nn cost function
[ cost, grad, pred_prob] = supervised_dnn_cost( params, ei, ...
    data_train, labels_train);






% %% setup minfunc options
% options = [];
% options.display = 'iter';
% options.maxFunEvals = 1e6;
% options.Method = 'lbfgs';
% 
% %% run training
% [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
%     params,options,ei, data_train, labels_train);

