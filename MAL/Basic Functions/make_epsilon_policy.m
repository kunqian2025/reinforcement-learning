function policy = make_epsilon_policy(Q_value, epsilon)
policy(1:length(Q_value)) = epsilon/length(Q_value);
%break the ties
[val,index] = max(Q_value);
[~,yy] = find(Q_value == val);
ties = size(yy);
if ties(2) > 1            
    index = ceil(rand(1)*ties(2));
    index = yy(index);
end
policy(index) = policy(index) + (1-epsilon);