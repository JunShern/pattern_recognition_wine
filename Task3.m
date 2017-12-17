wine_data = importdata('wine.data.csv');

% First need to separate data into training and testing
% 1-training, 2-test

total_size = size(wine_data,1);
train_data = zeros(total_size,size(wine_data,2));
test_data = zeros(total_size,size(wine_data,2));

count = 0;
count2 = 0;
for i = 1:total_size
     if wine_data(i,1) == 1
         count = count + 1;
         train_data(count,1:end) = wine_data(i, 1:end);
     elseif wine_data(i,1) == 2
         count2 = count2 + 1;
         test_data(count2, 1:end) = wine_data(i, 1:end);
     end
end

% Eliminate extra zeros

train_data( all(~train_data,2), : ) = [];   
test_data( all(~test_data,2), : ) = [];

% Extract labels and elimnate them from data matrices

train_label = train_data(:, 2);
test_label = test_data (:, 2);

train_data(:,1:2) = [];
test_data(:, 1:2) = [];

% Need to transpose to train the MATLAB built-in function for Neural Networks
train_data = train_data';
train_label = train_label';
test_data = test_data';
test_label = test_label';

binary_train_label = zeros(3,size(train_label,2)); 
binary_test_label = zeros(3,size(test_label,2)); 

count = 0;
for i = 1:size(train_label,2)
     if train_label(1,i) == 1
         count = count + 1;
         binary_train_label(1,count) = 1;
     elseif train_label(1,i) == 2
         count = count + 1;
         binary_train_label(2,count) = 1;
     elseif train_label(1,i) == 3
         count = count + 1;
         binary_train_label(3,count) = 1;
     end
end

count = 0;
for i = 1:size(test_label,2)
     if test_label(1,i) == 1
         count = count + 1;
         binary_test_label(1,count) = 1;
     elseif test_label(1,i) == 2
         count = count + 1;
         binary_test_label(2,count) = 1;
     elseif test_label(1,i) == 3
         count = count + 1;
         binary_test_label(3,count) = 1;
     end
end

%% Setting up the Neural Network

% Make starting random initial weights for neural networks not random
setdemorandstream(391418381);  

% Best classification percentage with 3 hidden layers of 15 neurons each
% [15 15 15]
% Default training function is 'trainscg' --> Scaled Conjugate Gradient
% 'traincgb' with [15, 15, 15] gives 95% accuracy
% 'traincgp' as well
net = patternnet([15, 15, 15], 'traincgb');
view(net);

[net,tr] = train(net,train_data,binary_train_label);
nntraintool;

plotperform(tr);

%% Testing the Neural Network

testY = net(test_data);
testIndices = vec2ind(testY);

%plotconfusion(binary_test_label,testY);

[c,cm] = confusion(binary_test_label,testY);

fprintf('Percentage Correct Classification  : %f%%\n', 100*(1-c));
fprintf('Percentage InCorrect Classification  : %f%%\n', 100*c);

plotroc(binary_test_label,testY);
