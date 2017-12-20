data = readtable('results_best_algs.csv');

algs = ['trainoss'; 'traingd '];
training_algs = cellstr(algs);
    
% Pre-allocate variables for speed
func_oss = zeros(31, 5); count_oss = 1;
func_gd = zeros(31, 5); count_gd = 1;

% Dictate which variable to sort on
sort_by_var = 5;

%% Partitioning

% Var1 = training algorithm
% Var2 = training time
% Var3 = test time
% Var4 = hidden layers
% Var5 = number of neurons
% Var6 = Accuracy
% Var7 = MSE

% Partition data into separate variables
for i=1:height(data)
    row = data(i, :);
    wine_data = [double(row.Var5), double(row.Var2), ...
                 double(row.Var3), double(row.Var6), ...
                 double(row.Var7)];
    ta = char(row.Var1);
    if strcmpi(ta, 'trainoss')
        func_oss(count_oss, :) = wine_data;
        count_oss = count_oss + 1;
    elseif strcmpi(ta, 'traingd')
        func_gd(count_gd, :) = wine_data;
        count_gd = count_gd + 1;
    end
end

%% Data Analysis

% For each of the arrays, find the two maximum and the two minimum
% To do this, form a matrix of structs

arr = [{func_oss};{func_gd}];

% Alternative: Matrix of label/value pairs
labls = zeros(2, 1);
labls(:) = [min(func_oss(:, sort_by_var)), min(func_gd(:, sort_by_var))];
[~, indices] = sort(labls);

% Best two algorithms are 1 and 2 of indices
disp(['Best algorithm is ', ...
      char(training_algs(indices(1)))]);
disp(['Second best algorithm is ', ...
      char(training_algs(indices(2)))]);

% disp(['Second worst algorithm is ', ...
%       char(training_algs(indices(end-1)))]);
% disp(['Worst algorithm is ', ...
%       char(training_algs(indices(end)))]);

%Plot Unmixed

% We now have the order of the best/worst algorithms and the associated
% data, so it's time to look at generating some plots. We want to see the
% hidden neurons on the x-axis. Plot the best one for now.

best_alg = cell2mat(arr(indices(1)));
second_alg = cell2mat(arr(indices(2)));
% second_worst_alg = cell2mat(arr(indices(end-1)));
% worst_alg = cell2mat(arr(indices(end)));

%% Plotting Processing
figure;
title('Mean Squared Error vs. Number of Neurons', 'interpreter', 'latex');
xlabel('Number of Neurons');
ylabel('Mean Squared Error');
% best = [indices(1), indices(2)];
hold on;
plot(best_alg(1:31, 1), best_alg(1:31, sort_by_var)/100);
plot(best_alg(94:124, 1), best_alg(94:124, sort_by_var)/100);
plot(second_alg(1:31, 1), second_alg(1:31, sort_by_var)/100);
plot(second_alg(94:124, 1), second_alg(94:124, sort_by_var)/100);
legend('show');
leg = legend('Single-layer traingbfg', '4-layer traingbfg', 'Single-layer traingcgp', '4-layer traincgp');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
set(leg,'FontSize', 16);
set(findall(gcf,'type','axes'),'fontsize', 20);
set(findall(gcf,'type','text'),'fontSize', 20);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('nets_best_algs2', '-dpng', '-r0');

% Set X limit to 100 to get interesting range
%ax.XLim = [0 100];
%print(strjoin({pic_path 'unmixed_best_limited'}, filesep), '-dpng');

%% Printing timing graphs

figure;
title('Training Time vs. Number of Neurons for traincgp', 'interpreter', 'latex');
xlabel('Number of Neurons');
ylabel('Training Time (s)');
hold on;
plot(second_alg(1:31, 1), second_alg(1:31, 2));
plot(second_alg(32:62,1), second_alg(32:62, 2));
plot(second_alg(63:93, 1), second_alg(63:93, 2));
plot(second_alg(94:124, 1), second_alg(94:124, 2));
legend('show');
leg = legend('1 Hidden Layer', '2 Hidden Layers', '3 Hidden Layers', '4 Hidden Layers');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
set(leg,'FontSize', 12);
set(findall(gcf,'type','axes'),'fontsize', 20);
set(findall(gcf,'type','text'),'fontSize', 20);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('timings_best_algs2', '-dpng', '-r0');
