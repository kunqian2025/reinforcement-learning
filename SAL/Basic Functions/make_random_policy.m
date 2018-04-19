function policy = make_random_policy(Q_value)
policy(1:length(Q_value)) = 1/length(Q_value);