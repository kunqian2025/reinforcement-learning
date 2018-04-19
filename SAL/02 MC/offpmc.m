function offpmc
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

% load('offpmc_policy.mat','-mat');
% q_value_or_policy2fig(Policy,env.locG,env.locO);
% return;


maxItr = 3000;%maximum iterations for ending one episode

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    gamma = 0.9; %discount factor
    epsilon = 0.8; %random action choice
    min_epsilon = 0.1;
    Q = zeros(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace));
    %The cumulative denominator of the weighted importance sampling formula
    C = zeros(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace)); 
    Policy = ones(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace))/length(env.actionSpace);
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    load('offpmc_q.mat','-mat');
    load('offpmc_c.mat','-mat');
    load('offpmc_policy.mat','-mat');
end

for itr=1:NUM_ITERATIONS 
    env.reset(env.locA); 
    if ~isTraining
        env.render();%display the moving environment
    end

    countActions = 0;%count how many actions in one iteration   
    done = false;
    
    matrix_episode= zeros(maxItr,4);
    agent_location = env.current_location;
    while ~done        
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1;
        if ~isTraining
            %choose an action based on target policy
            action = randsample(env.actionSpace,1,true,Policy(agent_location(1), agent_location(2),:));
            [next_location, reward, done] = env.step(action);
            env.render();%display the moving environment
            agent_location = next_location;
            continue;
        end
        %choose an action based on behavior policy 
        action = randsample(env.actionSpace,1,true,make_epsilon_policy(Q(agent_location(1), agent_location(2),:), max(epsilon^log(itr),min_epsilon)));
        
        [next_location, reward, done] = env.step(action);
        
        %create a matrix value
        matrix_episode(countActions,:) = [agent_location(1),agent_location(2),action,reward];
        
        agent_location = next_location;
    end
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
    if isTraining
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
        if done %reaches one of the terminal states
            G = 0;
            W = 1;
            for T=1:countActions
                t = countActions - T + 1;%backword
                location = matrix_episode(t,1:2);
                action =  matrix_episode(t,3);
                reward = matrix_episode(t,4);%R_t+1

                % Update the total reward since step t
                G = gamma * G + reward;

                % Update weighted importance sampling formula denominator
                C(location(1), location(2),action) =  C(location(1), location(2),action) + W;
                
                prob_a = make_epsilon_policy(Q(location(1), location(2),:), max(epsilon^log(itr),min_epsilon));%behavior policy
                
                % Update the action-value function using the incremental update formula (5.7)
                % This also improves our target policy which holds a reference to Q
                Q(location(1), location(2),action) = Q(location(1), location(2),action)...
                    +(W /C(location(1), location(2),action)) * (G - Q(location(1), location(2),action));

                Policy(location(1), location(2),:) = make_greedy_policy(Q(location(1), location(2),:)); 

                % If the action taken by the behavior policy is not the action 
                % taken by the target policy the probability will be 0 and we can break
                if Policy(location(1), location(2),action) ~= 1
                    break
                end
                %prob_a = make_epsilon_policy(Q(location(1), location(2),:), max(epsilon^log(itr),min_epsilon));%behavior policy
                W = W * 1/prob_a(action); % W * pi(a|s) / b(a|s)
            end  
        end
    end
end
if isTraining
    save('offpmc_q.mat','Q');
    save('offpmc_c.mat','C');
    save('offpmc_policy.mat','Policy');
    save('offpmc_iterationCount.mat','iterationCount');
    save('offpmc_reward.mat','rwd');
    figure,bar(iterationCount);
    figure,bar(rwd);
end
