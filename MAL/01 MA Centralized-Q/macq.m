function macq
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = true; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Basic Functions');
env = MAEnvironment;

alpha = 0.05; %learning rate settings 
gamma = 0.99; %discount factor
maxItr = 3000;%maximum iterations for ending one episode
central_estimator = LFAEstimator(env,alpha);

if isTraining
    NUM_ITERATIONS = 5000; %change this value to set max iterations
    max_epsilon = 0.8; %random action choice
    min_epsilon = 0.1;
    epsilon = max_epsilon:-(max_epsilon-min_epsilon)/NUM_ITERATIONS:min_epsilon;
    iterationCount(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    epsilon = 0.1; %random action choice
    load('weights.mat','-mat');
    central_estimator.set_weights(weights);
end
rwd(NUM_ITERATIONS) = 0;
for itr=1:NUM_ITERATIONS 
    env.reset(env.locA);
    if ~isTraining 
        env.render();%display the moving environment
    end
	state = env.current_location;
    
    countActions = 0;%count how many actions in one iteration  
    done = false;
    
    while ~done   
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1;         
        
        if ~isTraining
            values = central_estimator.predict(state, env.actionSpace);
            prob_a = make_epsilon_policy(values, epsilon);
            action = randsample(env.actionSpace,1,true,prob_a);
            a1_action = ceil(action/sqrt(length(env.actionSpace)));
            a2_action = action - sqrt(length(env.actionSpace))*(a1_action - 1);

            [next_state, reward, done] = env.step([a1_action a2_action]);
            rwd(itr) = rwd(itr) + reward;
            state = next_state;
            env.render();%display the moving environment
            continue;
        end
        values = central_estimator.predict(state, env.actionSpace);
        prob_a = make_epsilon_policy(values, epsilon(itr));
        action = randsample(env.actionSpace,1,true,prob_a);
        a1_action = ceil(action/sqrt(length(env.actionSpace)));
        a2_action = action - sqrt(length(env.actionSpace))*(a1_action - 1);

        [next_state, reward, done] = env.step([a1_action a2_action]);
        
        if done
            td_err = reward - values(action);
        else
            td_err = reward + gamma*max(central_estimator.predict(next_state, env.actionSpace)) - values(action);
        end
        central_estimator.update(state,action,td_err);
        rwd(itr) = rwd(itr) + reward;
        state = next_state;
    end
    fprintf('final location: %d, %d; %d, %d\n',state');
    fprintf('%d th iteration, %d actions taken, accumulated reward is %d.\n',itr,countActions,rwd(itr));
    if isTraining
        iterationCount(itr) = countActions;
    end
end
if isTraining
    weights = central_estimator.weights;
    save('weights.mat','weights');
    save('maq_iterationCount.mat','iterationCount');
    save('maq_reward.mat','rwd');
    histogram(iterationCount)
    histogram(rwd)
end