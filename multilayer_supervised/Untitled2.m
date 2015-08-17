    re = []; %result struct
    re.hy{1}.name = 'hidden layer 1 neuron number';
    re.hy{1}.value = hid1;
    re.hy{2}.name = 'hidden layer 1 neuron number';
    re.hy{2}.value = hid2;
    re.hy{3}.name = 'lambda';
    re.hy{3}.value = lambdam;
    
    re.test = result_test;
    re.train = result_train;