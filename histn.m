function [cnt, idx] = histn(X, E)
% n-dimensional histogram
%
% INPUT
%   - X: data matrix of size m x n, 
%        where m is the number of measurements and x \in R^n
%   - E: bin edgesges of size m x n
%        where m are the bin edges for dimension n
%
% OUTPUT
%   - cnt: quantized n-dim bincounts
%          i.e. |X(:,i)| s.t. edge(i) <= X(:,i) < edge(i+1)
%   - idx: index location of X in the quantized bins
%          points outside of boundary edges are assigned to
%          corresponding boundary bins

nd = size(X,2); idx = zeros(size(X));
sz = zeros(1,nd); edge_max=zeros(1,nd);

% iterate over nd dimensions
for d=1:nd
    Xd = X(:,d); [r,~]=find(E(:,d)); edges = E(r,d);
    % call histcounts for this dimension
    [~,idx(:,d)] = histc(Xd, edges, 1);
    sz(d) = length(edges)-1; %num centers
    edge_max(d)=max(edges);  %max edge
end

%post-process for zero idx (outside the bin range)
[r0,c0]=find(idx==0); idx_max=max(idx,[],1);

edgeu=zeros(length(r0),1);
for i=1:length(r0)
     edgeu(i)=(X(r0(i),c0(i))>=edge_max(c0(i)));
     if (edgeu(i)==1)
        idx(r0(i),c0(i))=sz(c0(i)); %assign to last bin
     else
        idx(r0(i),c0(i))=1; %assign to 1st bin
     end
end
    
% nd dim bincounts
cnt=accumarray(idx, 1, sz);

end
