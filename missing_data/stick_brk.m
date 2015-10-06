function [stick_length] = stick_brk(x_obs,alpha)
    [idx_nan]=isnan(x_obs); not_nan = ~idx_nan; num_not_nan = sum(not_nan);
    idx_not_nan = find(not_nan); x_sticks = zeros(num_not_nan,1);
    
    for i = 1:length(x_sticks)
        if (i == length(x_sticks))
            x_sticks(i) = betarnd(double(alpha(idx_not_nan(1))),double(sum(alpha(idx_not_nan(end)))));
            idx_not_nan(1)=[]; %remove current entry from not_nan list
        else
            x_sticks(i) = betarnd(double(alpha(idx_not_nan(1))),double(sum(alpha(idx_not_nan(2:end)))));
            idx_not_nan(1)=[]; %remove current entry from not_nan list
        end
    end
    stick_length = prod(ones(num_not_nan,1) - x_sticks);

