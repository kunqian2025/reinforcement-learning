function policy = make_greedy_policy(Q_value)
policy(1:length(Q_value)) = 0;
%break the ties
[val,index] = max(Q_value);
[~,yy] = find(Q_value == val);
ties = size(yy);
if ties(2) > 1            
    index = ceil(rand(1)*ties(2));
    index = yy(index);
end
policy(index) = 1;
