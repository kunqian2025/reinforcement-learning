function qLearning
close all;
clear; clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = true; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

load('qLearning_q.mat','-mat');
q_value_or_policy2fig(Q,env.locG,env.locO);
return;

alpha = 0.1; %learning rate settings 
gamma = 0.9; %discount factor
maxItr = 3000;%maximum iterations for ending one episode

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    epsilon = 0.8; %random action choice
    min_epsilon = 0.3;
    % build a state action matrix by finding all valid states from maze
    % we have 4 actions for each state.
    Q = rand(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace));
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    epsilon = 0.3; %random action choice
    load('qLearning_q.mat','-mat');
end

for itr=1:NUM_ITERATIONS 
    env.reset(env.locA); 
    if ~isTraining
        env.render();%display the moving environment
    end
    
    countActions = 0;%count how many actions in one iteration   
    reward = 0;
    done = false;
    agent_location = env.current_location;
    while ~done   
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1;
        
       if ~isTraining
            prob_a = make_epsilon_policy(Q(agent_location(1), agent_location(2),:), epsilon);
            action = randsample(env.actionSpace,1,true,prob_a);
            [next_location, reward, done] = env.step(action);
            env.render();%display the moving environment
            agent_location = next_location;
            continue;
        end
        
        prob_a = make_epsilon_policy(Q(agent_location(1), agent_location(2),:), max(epsilon^log(itr),min_epsilon));
        action = randsample(env.actionSpace,1,true,prob_a);
        
        [next_location, reward, done] = env.step(action);
        
        % update information for robot in Q for later use
        Q(agent_location(1), agent_location(2),action) = Q(agent_location(1), agent_location(2),action) + ...
            alpha*(reward+gamma*max(Q(next_location(1), next_location(2),:)) - Q(agent_location(1), agent_location(2),action));
        
        agent_location = next_location;
    end
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
    if isTraining
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
    end
end
if isTraining
    save('qLearning_q.mat','Q');
    save('qLearning_iterationCount.mat','iterationCount');
    save('qLearning_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end
% load('qLearning_iterationCount.mat','-mat')
% figure,bar(iterationCount)
% load('qLearning_reward.mat','-mat')
% figure,bar(rwd)