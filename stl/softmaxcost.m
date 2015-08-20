function [ output_args ] = softmaxcost( input_args )
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


end

