function [data] = generate_data()

N = 1e3;        %words (dictionary size)
D = 10;         %documents
K = randi(5)+2; %topics

data = zeros(N,D);
for k=1:K
    pi = dirrnd(ones(D,1)*0.05);    
    
    istart = floor(N/K*(k-1)) + 1;
    istop = floor(N/K*k);    
    
    numWords = randi(N,istop-istart+1,1);
    data(istart:istop,:) = mnrnd(numWords,pi);
end
%data = data';
data = sparse(data);