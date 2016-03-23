%% self-avoiding random walks using randomized monte-carlo algorithm
%
%  Assumes equiprobable moves to a set of unoccupied locations
%  Uses sequential importance sampling to correct for bias with
%  weights = p(x_0,...,x_n) / q(x_0,...,x_n). Assuming independent
%  moves, choose the proposal q(x_0,...,x_i) = 1/prod_i d_{i},
%  where d_{i-1} is the number of unoccupied neighbors of x_{i-1} and
%  q(x_i|x_0,...,x_{i-1}) = 1/d_{i-1}.

clear all; close all;

%% monte-carlo randomized algorithm

n=150; %number of steps in a random walk
num_iter = 1e2; %number of iterations for averaging results
moves = [0 1; 0 -1; -1 0; 1 0]; %2-D moves

%random walk statistics
square_dist = zeros(num_iter,1); weights = zeros(num_iter,1);

for iter=1:num_iter  
    trial = 0; i=1;
    %iterate until we have a non-crossing random walk
    while (i~=n) 
    
        %init
        X=0; Y=0; weight = 1;
        lattice = zeros(2*n+1, 2*n+1);
        lattice(n+1,n+1) = 1;
        path = [0 0];     
        xx = n + 1 + X;
        yy = n + 1 + Y;    
        fprintf('iter: %d, trial: %d\n', iter, trial);
    
        for i=1:n                    
           up    = lattice(xx,yy+1);
           down  = lattice(xx,yy-1);
           left  = lattice(xx-1,yy);
           right = lattice(xx+1,yy);
       
           %compute available directions
           neighbors = [1 1 1 1] - [up, down, left, right];
       
           %avoid self-loops
           if (sum(neighbors)==0), i=1; break; end

           %compute importance weights: d0 x d1 x ... x d_{n-1}
           %note: grows as O(4^n), exponential with the number of steps n
           weight = weight * sum(neighbors);
           
           %sample a move direction
           direction = find(rand < (cumsum(neighbors)/sum(neighbors)),1,'first');
       
           %updated local coordinates
           X = X + moves(direction,1);
           Y = Y + moves(direction,2);
       
           %assign random walk events
           %U = rand;
           %if (U < 1/2)
           %   X = X + 2*(U < 1/4) - 1; %{Left, Right}
           %else
           %   Y = Y + 2*(U < 3/4) - 1; %{Up,Down}
           %end
                   
           %store sampled path
           path_new = [X Y];
           path = [path; path_new];

           %update grid coordinates
           xx = n + 1 + X;
           yy = n + 1 + Y;               
           lattice(xx,yy) = 1;       
        end
        trial=trial+1;    
    end
    %compute square extension
    square_dist(iter) = X^2 + Y^2;                     
    %store importance weights
    weights(iter) = weight;
end

%compute mean square extension
mean_square_dist = mean(weights.*square_dist)/mean(weights);
fprintf('mean square dist: %d\n', mean_square_dist);

%% generate plots
figure;
clf; hold on; axis([-n n -n n]);
for i=1:n-1
    line([path(i,1),path(i+1,1)],[path(i,2),path(i+1,2)],'linewidth',2.0);  
end
title('random walk with no overlaps'); xlabel('x'); ylabel('y');

figure;
hist(square_dist);
title('square distance of the random walk'); xlabel('square distance (X^2 + Y^2)');

figure;
semilogy(weights);
title('random walk importance weights'); 
xlabel('number of iterations'); ylabel('weights');
