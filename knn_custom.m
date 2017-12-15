function indices = knn_custom(X,Y,k,dist_func)
% Rows are elements
% Columns are features
indices = zeros(size(Y,1),size(X,1));
for ind_y = 1:size(Y,1)
    A = normr(Y(ind_y,:));
    B = normr(X);
    
    vals = dist_func(A,B);
    inds = 1:size(vals,2);
    neighbors = vertcat(vals, inds);
    neighbors = sortrows(neighbors',1)';
    %[~, closest_ind] = min(dist_func(A,B));
    indices(ind_y,:) = neighbors(2,:);
end
% Take k nearest neighbors
indices = indices(:,1:k);

end