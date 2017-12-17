function [cluster_indices, centroid_data] = kmeans_custom(X,k,dist_func,max_iter)
if nargin < 4
    max_iter = 100; % Default maximum number of iterations
end
%X = normr(X); % Cannot norm; we want centroids to preserve scale
% Rows are elements
% Columns are features
cluster_indices = zeros(size(X,1),1);
centroid_data= init_centroids(X,k);
for i = 1:max_iter
    cluster_indices = assign_clusters(X,centroid_data,dist_func);
    centroid_data = new_centroids(X,cluster_indices,k);
end
end

function centroid_data = init_centroids(X,k)
random_indices = randperm(size(X,1))';
centroid_data = X(random_indices(1:k),:);
end

function centroid_data = new_centroids(X,cluster_indices,k)
centroid_data = zeros(k,size(X,2));
for i=1:k
    cluster_members = X(cluster_indices==i,:);
    cluster_mean = mean(cluster_members,1);
    centroid_data(i,:) = cluster_mean;
end 
end

function cluster_indices = assign_clusters(X,centroid_data,dist_func);
% POTENTIALLY SLOW BOTTLENECK
cluster_indices = zeros(size(X,1),1);
for i=1:size(X,1)
    datapoint = X(i,:);
    distances = dist_func(datapoint,centroid_data);
    [~,cluster_indices(i,:)] = min(distances);
end
end