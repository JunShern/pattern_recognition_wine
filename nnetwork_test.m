function [accuracy,time_train,time_test,error] = nnetwork_test(training,l_training,test,l_test,neurons,n_layers,net_type,func_train)
% Function to create a massive test of Neural Networks with the provided data
% neurons: number of neurons used in the hidden layers
% n_layers: number of hidden layers
% net_type: different in-built Neural Network MATLAB functions
% func_train: different training algorithms

% Need to first do an array for the hidden layers

layers = zeros(1,n_layers);
for i = 1:n_layers
   layers(i) = neurons; 
end

% Selecting which MATLAB built-in function to use

if strcmpi(net_type, 'patternnet')
    net = patternnet(layers, func_train);
else
    disp('Network type not recognized')
    return
end

net.performFcn = 'mse';     % Set the performance function to mean squared error

% Disable the UI window
net.trainParam.showWindow = 0;

% Train the Neural Network

% Make starting random initial weights for neural networks not random
setdemorandstream(391418381);

tic();
net = train(net,training, l_training);
time_train = toc();

% Run test data through network

tic();
outputs = net(test);
time_test = toc();

%Compare targets to outputs

[c,cm] = confusion(l_test,outputs);

accuracy = 100*(1-c);
error = 100*c; 

% correct = round(outputs) == l_test;
% accuracy = sum(correct) / length(l_test);

% % Get squared error sum as an error
% err = gsubtract(outputs, l_test);
% error = err .^ 2;

end

