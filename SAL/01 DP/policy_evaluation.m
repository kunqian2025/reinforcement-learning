function V = policy_evaluation(env, Policy, V, gamma, theta)
countItr = 0;
while true
	delta = 0;
	%perform backup for all states
	for i=1:env.sizeMap(1)
		for j=1:env.sizeMap(2)
			if [i,j] == env.locG
				continue;
			end
			if [i,j] == env.locO
				continue;
			end
			v = 0;
			for action=1:length(env.actionSpace)
				[prob, next_state, reward] = env.possible_next_state([i,j], action);
				for possi=1:length(env.actionPossibilities)
					v = v + Policy(i,j,action) * prob(possi) * (reward(possi) + gamma * V(next_state(possi,1),next_state(possi,2)));
				end
			end
			delta = max(delta, abs(v - V(i,j)));
			V(i,j) = v;
		end
	end
	countItr = countItr + 1;
	fprintf('    %d th evaluation iteration, final error is %d.\n',countItr,delta);
	if delta < theta
		break;
	end
end