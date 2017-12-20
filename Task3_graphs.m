wine_data = importdata('wine.data.csv');

% Make different arrays with variables in order to test for different
% parameters

neurons = [1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,100,110,120,130,140,150,160,170,180,190,200];
funcs = ['traingd '; 'traincgp'];
hidden_layers = [1,2,3,4];
ntypes = 'patternnet';
net_types = cellstr(ntypes);
training_funcs = cellstr(funcs);

% Need to store everything into something (a CSV file for example)
handle = fopen('results_best_algs3.csv', 'w'); 
headings = ['NetworkType,TrainingAlgorithm,'...
            'HiddenLayers,HiddenNeurons,TimetoTrain,TimetoTest,'...
            'Accuracy,SumSquaredError,MeanSquaredError', ...
            sprintf('\n')];
fwrite(handle,headings);
fclose(handle);

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

%% Trying out many different parameters

for nt = 1:length(net_types)
    for ta = 1:length(training_funcs)
        for hl = 1:length(hidden_layers)
            for ne = 1:length(neurons)
            func = deblank(char(training_funcs(ta)));
            ntype = deblank(char(net_types(nt)));
            % Calling function to use Neural Network
            [accuracy, time_train, time_test, error] = nnetwork_test(train_data, binary_train_label,...
                test_data, binary_test_label, neurons(ne), hidden_layers(hl), ntype, func);
            disp(['Type: ', ntype, '; Func: ', func, '; Layers: ',...
                          num2str(hidden_layers(hl)), '; Neurons: ', ...
                          num2str(neurons(ne)), '; Time_train: ',...
                          num2str(time_train), '; Accuracy: ' num2str(accuracy) '%', '; Error: ', ...
                          num2str(sum(error)), '%']);
            data = [func, ',' num2str(time_train), ',', num2str(time_test), ',', ... 
                num2str(hidden_layers(hl)), ',', num2str(neurons(ne)), ',', num2str(accuracy), ',', ... 
                num2str(sum(error)), ',', num2str(sum(error)/length(error)), ...
                ',', sprintf('\n')];
            handle = fopen('results_best_algs3.csv', 'a');
            fwrite(handle,data);
            fclose(handle);
            end
        end
    end
end