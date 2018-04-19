function PI
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = true; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

if ~isTraining
    load('PI_P.mat','-mat');
    q_value_or_policy2fig(Policy,env.locG,env.locO);
    return;
end

gamma = 0.9;% - discount factor
theta = 0.0001;% - max error for stopping iteration
Policy(1:env.sizeMap(1), 1:env.sizeMap(2),1) = 0.1;
Policy(1:env.sizeMap(1), 1:env.sizeMap(2),2) = 0.2;
Policy(1:env.sizeMap(1), 1:env.sizeMap(2),3) = 0.3;
Policy(1:env.sizeMap(1), 1:env.sizeMap(2),4) = 0.4;

% Policy = ones(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace))/length(env.actionSpace);
Policy(env.locG(1),env.locG(2),:) = 0;
Policy(env.locO(1),env.locO(2),:) = 0;
V = zeros(env.sizeMap(1),env.sizeMap(2));

countItr = 0;
timeStart = clock;
while true
    countItr = countItr + 1;
	fprintf('%d th policy iteration.\n',countItr);
    V = policy_evaluation(env, Policy, V, gamma, theta);
    isStable = true;
    for i=1:env.sizeMap(1)
		for j=1:env.sizeMap(2)
            if [i,j] == env.locG
				continue;
            end
            if [i,j] == env.locO
				continue;
            end
            %find the best action accroding to the policy
            [~,old_action] = max(Policy(i,j,:));
            
            %one step lookahead to see which action is the best 
			v(1:length(env.actionSpace)) = 0;
            for action=1:length(env.actionSpace)
				[prob, next_state, reward] = env.possible_next_state([i,j], action);
                
				for possi=1:length(env.actionPossibilities)
					v(action) = v(action) + prob(possi) * (reward(possi) + gamma * V(next_state(possi,1),next_state(possi,2)));
				end
            end
            [~,best_action] = max(v);
            
            if old_action ~= best_action
                isStable = false;
                Policy(i,j,:) = 0;%policy improvement
                Policy(i,j,best_action) = 1;
            end

		end
    end
    if isStable
        timeEnd = clock;
        simulationTime = sum([timeEnd - timeStart].*[0 0 0 3600 60 1]);
        save('PI_V.mat','V'); 
        save('PI_P.mat','Policy'); 
        save('PI_simulationTime.mat','simulationTime'); 
        break;
    end
end

