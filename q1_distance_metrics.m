%% Read data into training and testing sets
[train,test] = importWineFile('wine.data.csv');
train_data = train(:,2:end);
test_data = test(:,2:end);
train_labels = train(:,1);
test_labels = test(:,1);

% Parameters
k_arr = [1:4:size(train_data,1)];
% chisquare = @(A,B)pdist2_custom(A,B,'chisq');
chisquaredist = @(x,Z)sqrt((bsxfun(@minus,x,Z).^2)*w);
earthmovers = @(A,B)pdist2_custom(A,B,'emd');
kullbackleibler = @(A,B)KLDiv(A,B);
distance_metrics = ["cityblock","euclidean","chebychev","cosine","correlation","mahalanobis","chisquare","kullbackleibler","earthmovers"]; % "minkowski","seuclidean"

% STILL NEED TO IMPLEMENT: Intersection, Kullback-Leibler divergence,
% Jenses-Shanon divergence, Chi square, Quadratic form, Earth movers

%%
errors = zeros(size(distance_metrics,2), size(k_arr,2));
for ind_k = 1:size(k_arr,2)
    for ind_d = 1:size(distance_metrics,2)
        % Parameters
        K = k_arr(ind_k);
        %nsmethod = 'kdtree' % 'exhaustive'
        dist = char(distance_metrics(ind_d))

        if strcmp(dist,'chisquare')
            neighbor_indices = knnsearch(train_data, test_data, 'K', K, 'distance', chisquare);
        elseif strcmp(dist,'earthmovers')
            neighbor_indices = knnsearch(train_data, test_data, 'K', K, 'distance', earthmovers);
        elseif strcmp(dist,'kullbackleibler')
            neighbor_indices = knnsearch(train_data, test_data, 'K', K, 'distance', kullbackleibler);
        else
            neighbor_indices = knnsearch(train_data, test_data, 'K', K, 'distance', dist);
        end
        
        neighbor_labels = zeros(size(neighbor_indices));
        for n = 1:size(neighbor_indices,2)
            neighbor_labels(:,n) = train_labels(neighbor_indices(:,n));
        end
        predicted_labels = mode(neighbor_labels,2);
        errors(ind_d,ind_k) = sum(predicted_labels ~= test_labels) / size(test_labels,1);
    end
end

%% Plot
figure('position', [0 0 1280 800]);
hold on;
for i = 1:size(distance_metrics,2)
    err = errors(i,:);
    plot(k_arr, err, 'linewidth', 3)
end
title('k-Nearest-Neighbors with different Distance Metrics', 'interpreter', 'latex')
xlabel('k');
ylabel('Classification Error');
grid;
leg = legend(distance_metrics, 'Location','southeast');
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize', 32);

saveas(gcf,'knn_errors.png')
