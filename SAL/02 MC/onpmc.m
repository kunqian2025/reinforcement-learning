function onpmc
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;


% load('onpmc_policy.mat','-mat');
% q_value_or_policy2fig(Policy,env.locG,env.locO);
% return;


maxItr = 3000;%maximum iterations for ending one episode

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    gamma = 0.9; %discount factor
    epsilon = 0.8; %random action choice
    min_epsilon = 0.1;
    Q = zeros(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace));
    Returns = zeros(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace),2);%save both sum and count
    Policy = ones(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace))/length(env.actionSpace);
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    load('onpmc_q.mat','-mat');
    load('onpmc_returns.mat','-mat');
    load('onpmc_policy.mat','-mat');
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
            action = randsample(env.actionSpace,1,true,Policy(agent_location(1),agent_location(2),:));
            [next_location, reward, done] = env.step(action);
            env.render();%display the moving environment
            agent_location = next_location;
            continue;
        end
        
        action = randsample(env.actionSpace,1,true,Policy(agent_location(1),agent_location(2),:));
        
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
            for t=1:countActions
                not_visited = true;
                for b_t=1:t-1 %check the state action pair before t
                    if isequal(matrix_episode(b_t,1:3),matrix_episode(t,1:3)) %check state action pair
                        not_visited = false;
                        break;
                    end
                end 
                if not_visited
                    location = [matrix_episode(t,1) matrix_episode(t,2)];
                    action = matrix_episode(t,3);   
                    G = 0;
                    for a_t = t:countActions
                        G = G + matrix_episode(a_t,4)*(gamma^(a_t-t));%Sum up all rewards since the first occurance
                        %R_t+1 is the reward of s_t, a_t
                    end

                    %Calculate average return for this state action over all sampled episodes
                    Returns(location(1), location(2),action,1) = Returns(location(1), location(2),action,1) + G;
                    Returns(location(1), location(2),action,2) = Returns(location(1), location(2),action,2) + 1;
                    Q(location(1), location(2),action) = Returns(location(1), location(2),action,1)/Returns(location(1), location(2),action,2); 

                    policy = make_epsilon_policy(Q(location(1), location(2),:), max(epsilon^log(itr),min_epsilon));
                    Policy(location(1),location(2),:) = policy;
                end
            end   
        end
    end
end
if isTraining
    save('onpmc_q.mat','Q');
    save('onpmc_returns.mat','Returns');
    save('onpmc_policy.mat','Policy');
    save('onpmc_iterationCount.mat','iterationCount');
    save('onpmc_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end
