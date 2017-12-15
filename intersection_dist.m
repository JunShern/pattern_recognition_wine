function dist = intersection_dist(A,B)
if (size(A,1) > 1)
    A
    B
    error('Expecting a single row vector for A, and m row vectors of the same length for B');
end

% A = normr(A);
% B = normr(B);
dist = zeros(size(B,1),1);
for i=1:size(B,1)
    b = B(i,:);
    mins = min(vertcat(b,A),[],1);
    dist(i,1) = sum(mins, 2);
end
dist = dist';
end