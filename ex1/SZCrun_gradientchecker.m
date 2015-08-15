%Sun Zhicheng
%test grad_checker, have to kown train already;
clc;

%parameters
theta0 = rand(size(theta))*0.001;
num_checks=50;

%test gradient
average_error = grad_check(@softmax_regression_vec, ...
    theta0, num_checks, train.X, train.y);