% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];
Ctrl = 3; %0 train; 1. hyperparameter 2. many hyperpara-2 3.many hyperpara-3

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
ei.layer_sizes = [200, 10, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.00003;
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
    hid_single_m = 100:50:300; %200 seems to be best value 0.985
    ei.layer_sizes(1) = 200;
    
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
elseif Ctrl ==2
    %% many hyperparameter test -2
    hid1 = 150:50:250;  %250 50 seems to be best
    hid2 = [50 100 150 200];
    lambdam = [0.00001 0.00003 0.0001]; %0.00003 seems to be best    
    
    re = []; %result struct
    re.hy{1}.name = 'hidden layer 1 neuron number';
    re.hy{1}.value = hid1;
    re.hy{2}.name = 'hidden layer 2 neuron number';
    re.hy{2}.value = hid2;
    re.hy{3}.name = 'lambda';
    re.hy{3}.value = lambdam;
    %record:1.para 2.test_error 3.train_error
    re.test = zeros(length(hid1),length(hid2),length(lambdam));
    re.train = zeros(length(hid1),length(hid2),length(lambdam));
    
    for i = 1:length(hid1)
        for j = 1:length(hid2)
            for k = 1:length(lambdam)
                ei.layer_sizes(1) = hid1(i);
                ei.layer_sizes(2) = hid2(j);
                ei.lambda = lambdam(k);
                stack = initialize_weights(ei);
                params = stack2params(stack);
                %tic;
                [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                    params,options,ei, data_train, labels_train);
                %result(i,4)=toc; 

                [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
                [~,pred] = max(pred);
                acc_test = mean(pred'==labels_test);
                re.test(i,j,k)=acc_test;

                [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
                [~,pred] = max(pred);
                acc_train = mean(pred'==labels_train);
                re.train(i,j,k)=acc_train;
            end
        end
    end
    %disp(result_test);
elseif Ctrl ==3
    %% many hyperparameter test -3 
    hid1 = 200:50:500; %250 50 seems to be best
    hid2 = [10 20 40 60];
    ei.lambda = 0.00003;
    
    %record:1.para 2.test_error 3.train_error
    re = 0; %result struct
    re.hy1.name = 'hidden layer 1 neuron number';
    re.hy1.value = hid1;
    re.hy2.name = 'hidden layer 2 neuron number';
    re.hy2.value = hid2;
    re.test = zeros(length(hid1),length(hid2));
    re.train = zeros(length(hid1),length(hid2));
    
    for i = 1:length(hid1)
        for j = 1:length(hid2)
            ei.layer_sizes(1) = hid1(i);
            ei.layer_sizes(2) = hid2(j);
            stack = initialize_weights(ei);
            params = stack2params(stack);
            
            [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
                params,options,ei, data_train, labels_train);

            [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
            [~,pred] = max(pred);
            acc_test = mean(pred'==labels_test);
            re.test(i,j)=acc_test;

            [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
            [~,pred] = max(pred);
            acc_train = mean(pred'==labels_train);
            re.train(i,j)=acc_train;
        end
    end
    disp(result_test);    
end


    