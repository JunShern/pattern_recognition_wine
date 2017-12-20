data = readtable('results7.csv');

algs = ['trainlm '; 'trainbr '; 'trainbfg'; 'trainrp '; ...
        'trainscg'; 'traincgb'; 'traincgf'; 'traincgp'; ...
        'trainoss'; 'traingdx'; 'traingdm'; 'traingd '];
training_algs = cellstr(algs);
    
% Pre-allocate variables for speed
func_lm = zeros(16, 5);  count_lm = 1;
func_br = zeros(16, 5);  count_br = 1;
func_bfg = zeros(16, 5); count_bfg = 1;
func_rp = zeros(16, 5);  count_rp = 1;
func_scg = zeros(16, 5); count_scg = 1;
func_cgb = zeros(16, 5); count_cgb = 1;
func_cgf = zeros(16, 5); count_cgf = 1;
func_cgp = zeros(16, 5); count_cgp = 1;
func_oss = zeros(16, 5); count_oss = 1;
func_gdx = zeros(16, 5); count_gdx = 1;
func_gdm = zeros(16, 5); count_gdm = 1;
func_gd = zeros(16, 5);  count_gd = 1;

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
    if strcmpi(ta, 'trainlm')
        func_lm(count_lm, :) = wine_data;
        count_lm = count_lm + 1;
    elseif strcmpi(ta, 'trainbr')
        func_br(count_br, :) = wine_data;
        count_br = count_br + 1;
    elseif strcmpi(ta, 'trainbfg')
        func_bfg(count_bfg, :) = wine_data;
        count_bfg = count_bfg + 1;
    elseif strcmpi(ta, 'trainrp')
        func_rp(count_rp, :) = wine_data;
        count_rp = count_rp + 1;
    elseif strcmpi(ta, 'trainscg')
        func_scg(count_scg, :) = wine_data;
        count_scg = count_scg + 1;
    elseif strcmpi(ta, 'traincgb')
        func_cgb(count_cgb, :) = wine_data;
        count_cgb = count_cgb + 1;
    elseif strcmpi(ta, 'traincgf')
        func_cgf(count_cgf, :) = wine_data;
        count_cgf = count_cgf + 1;
    elseif strcmpi(ta, 'traincgp')
        func_cgp(count_cgp, :) = wine_data;
        count_cgp = count_cgp + 1;
    elseif strcmpi(ta, 'trainoss')
        func_oss(count_oss, :) = wine_data;
        count_oss = count_oss + 1;
    elseif strcmpi(ta, 'traingdx')
        func_gdx(count_gdx, :) = wine_data;
        count_gdx = count_gdx + 1;
    elseif strcmpi(ta, 'traingdm')
        func_gdm(count_gdm, :) = wine_data;
        count_gdm = count_gdm + 1;
    elseif strcmpi(ta, 'traingd')
        func_gd(count_gd, :) = wine_data;
        count_gd = count_gd + 1;
    end
end

%% Data Analysis

% For each of the arrays, find the two maximum and the two minimum
% To do this, form a matrix of structs

arr = [{func_lm };
             {func_br };
             {func_bfg};
             {func_rp };
             {func_scg};
             {func_cgb};
             {func_cgf};
             {func_cgp};
             {func_oss};
             {func_gdx};
             {func_gdm};
             {func_gd };];

% Alternative: Matrix of label/value pairs
labls = zeros(12, 1);
labls(:) = [min(func_lm (:, sort_by_var)), ...
                 min(func_br (:, sort_by_var)), ...
                 min(func_bfg(:, sort_by_var)), ...
                 min(func_rp (:, sort_by_var)), ...
                 min(func_scg(:, sort_by_var)), ...
                 min(func_cgb(:, sort_by_var)), ...
                 min(func_cgf(:, sort_by_var)), ...
                 min(func_cgp(:, sort_by_var)), ...
                 min(func_oss(:, sort_by_var)), ...
                 min(func_gdx(:, sort_by_var)), ...
                 min(func_gdm(:, sort_by_var)), ...
                 min(func_gd (:, sort_by_var))];
[~, indices] = sort(labls);

% Best two algorithms are 1 and 2 of indices
disp(['Best algorithm is ', ...
      char(training_algs(indices(1)))]);
disp(['Second best algorithm is ', ...
      char(training_algs(indices(2)))]);

disp(['Second worst algorithm is ', ...
      char(training_algs(indices(end-1)))]);
disp(['Worst algorithm is ', ...
      char(training_algs(indices(end)))]);

% We now have the order of the best/worst algorithms and the associated
% data, so it's time to look at generating some plots. We want to see the
% hidden neurons on the x-axis. Plot the best one for now.

best_alg = cell2mat(arr(indices(1)));
second_alg = cell2mat(arr(indices(2)));
second_worst_alg = cell2mat(arr(indices(end-1)));
worst_alg = cell2mat(arr(indices(end)));

%% Plotting Processing
figure;
title('Mean Squared Error vs. Number of Neurons', 'interpreter', 'latex');
xlabel('Number of Neurons');
ylabel('Mean Squared Error');
hold on;
plot(func_bfg(1:16, 1), func_bfg(1:16, sort_by_var)/100);
plot(func_br(1:16, 1), func_br(1:16, sort_by_var)/100);
plot(func_cgb(1:16, 1), func_cgb(1:16, sort_by_var)/100);
plot(func_cgf(1:16, 1), func_cgf(1:16, sort_by_var)/100);
plot(func_cgp(1:16, 1), func_cgp(1:16, sort_by_var)/100);
plot(func_gd(1:16, 1), func_gd(1:16, sort_by_var)/100);
plot(func_gdm(1:16, 1), func_gdm(1:16, sort_by_var)/100);
plot(func_gdx(1:16, 1), func_gdx(1:16, sort_by_var)/100);
plot(func_lm(1:16, 1), func_lm(1:16, sort_by_var)/100);
plot(func_oss(1:16, 1), func_oss(1:16, sort_by_var)/100);
plot(func_rp(1:16, 1), func_rp(1:16, sort_by_var)/100);
plot(func_scg(1:16, 1), func_scg(1:16, sort_by_var)/100);
legend('show');
leg = legend('trainbfg', 'trainbr', 'traincgb', 'traincgf', 'traincgp', 'traingd', ...
    'traingdm', 'traingdx', 'trainlm', 'trainoss', 'trainrp', 'trainscg');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
% Formatting for Latex
set(leg,'FontSize', 14);
set(findall(gcf,'type','axes'),'fontsize', 20);
set(findall(gcf,'type','text'),'fontSize', 20);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('graph_nets2', '-dpng', '-r0');

%% Plot Timings

figure;
title('Training Time vs. Number of Neurons', 'interpreter', 'latex');
xlabel('Number of Neurons');
ylabel('Training Time (s)');
hold on;
plot(func_bfg(1:16, 1), func_bfg(1:16, 2));
plot(func_br(1:16, 1), func_br(1:16, 2));
plot(func_cgb(1:16, 1), func_cgb(1:16, 2));
plot(func_cgf(1:16, 1), func_cgf(1:16, 2));
plot(func_cgp(1:16, 1), func_cgp(1:16, 2));
plot(func_gd(1:16, 1), func_gd(1:16, 2));
plot(func_gdm(1:16, 1), func_gdm(1:16, 2));
plot(func_gdx(1:16, 1), func_gdx(1:16, 2));
plot(func_lm(1:16, 1), func_lm(1:16, 2));
plot(func_oss(1:16, 1), func_oss(1:16, 2));
plot(func_rp(1:16, 1), func_rp(1:16, 2));
plot(func_scg(17:32, 1), func_scg(17:32, 2));
legend('show');
leg = legend('trainbfg', 'trainbr', 'traincgb', 'traincgf', 'traincgp', 'traingd', ...
    'traingdm', 'traingdx', 'trainlm', 'trainoss', 'trainrp', 'trainscg');
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
set(leg,'FontSize', 12);
set(findall(gcf,'type','axes'),'fontsize', 18);
set(findall(gcf,'type','text'),'fontSize', 18);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('graph_nets_timings2', '-dpng', '-r0');