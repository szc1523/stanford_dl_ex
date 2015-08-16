% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];
Ctrl = 1; %0 train; 1 hyperparameter

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
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.002;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';


if Ctrl==0    
    %% run training once
    tic;
    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
        params,options,ei, data_train, labels_train);
    fprintf('Optimization took %f seconds.\n', toc);

    %% compute accuracy on the test and train set
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
    [~,pred] = max(pred);
    acc_test = mean(pred'==labels_test);
    fprintf('test accuracy: %f\n', acc_test);

    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
    [~,pred] = max(pred);
    acc_train = mean(pred'==labels_train);
    fprintf('train accuracy: %f\n', acc_train);


elseif Ctrl ==1
    %% hyperparameter test
    lambdam = [0 0.00001 0.00003 0.0001]; %0.00003 seems to be best
    ei.lambda = 0.00003;
    hid_single_m = 100:50:300;
    
    curpara = hid_single_m; %its current para
    %record:1.para 2.test_error 3.train_error 4.time
    result = zeros(length(curpara), 4);
    result(:,1) = curpara';
    
    for i = 1:length(curpara)
        ei.layer_sizes(1) = curpara(i);
        stack = initialize_weights(ei);
        params = stack2params(stack);
        tic;
        [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
            params,options,ei, data_train, labels_train);
        result(i,4)=toc; 

        [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
        [~,pred] = max(pred);
        acc_test = mean(pred'==labels_test);
        result(i,2)=acc_test;

        [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
        [~,pred] = max(pred);
        acc_train = mean(pred'==labels_train);
        result(i,3)=acc_train;
    end
    disp(result);
end


    