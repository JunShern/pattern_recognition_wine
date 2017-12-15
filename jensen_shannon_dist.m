function dist = jensen_shannon_dist(A,B)
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
    dist(i,1) = jensen_shannon_divergence(b,A);
end
dist = dist';
end