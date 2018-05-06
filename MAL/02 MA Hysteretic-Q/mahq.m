function mahq
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = true; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Basic Functions');
env = MAEnvironment;

alpha = 0.05; %learning rate settings 
beta = 0.005;
gamma = 0.99; %discount factor
maxItr = 3000;%maximum iterations for ending one episode
a1_estimator = LFAEstimator(env,alpha,beta);
a2_estimator = LFAEstimator(env,alpha,beta);

if isTraining
    NUM_ITERATIONS = 5000; %change this value to set max iterations
    max_epsilon = 0.8; %random action choice
    min_epsilon = 0.1;
    epsilon = max_epsilon:-(max_epsilon-min_epsilon)/NUM_ITERATIONS:min_epsilon;
    iterationCount(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    epsilon = 0.1; %random action choice
    load('a1_weights.mat','-mat');
    load('a2_weights.mat','-mat');
    a1_estimator.set_weights(a1_weights);
    a2_estimator.set_weights(a2_weights);
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
            a1_values = a1_estimator.predict(state, env.actionSpace);
            prob_a = make_epsilon_policy(a1_values, epsilon);
            a1_action = randsample(env.actionSpace,1,true,prob_a);

            a2_values = a2_estimator.predict(state, env.actionSpace);
            prob_a = make_epsilon_policy(a2_values, epsilon);
            a2_action = randsample(env.actionSpace,1,true,prob_a);

            [next_state, reward, done] = env.step([a1_action a2_action]);
            rwd(itr) = rwd(itr) + reward;
            state = next_state;
            env.render();%display the moving environment
            continue;
        end
        a1_values = a1_estimator.predict(state, env.actionSpace);
        prob_a = make_epsilon_policy(a1_values, epsilon(itr));
        a1_action = randsample(env.actionSpace,1,true,prob_a);
        
        a2_values = a2_estimator.predict(state, env.actionSpace);
        prob_a = make_epsilon_policy(a2_values, epsilon(itr));
        a2_action = randsample(env.actionSpace,1,true,prob_a);
        
        [next_state, reward, done] = env.step([a1_action a2_action]);
        
        if done
            td_err1 = reward - a1_values(a1_action);
            td_err2 = reward - a2_values(a2_action);
        else
            td_err1 = reward + gamma*max(a1_estimator.predict(next_state, env.actionSpace)) - a1_values(a1_action);
            td_err2 = reward + gamma*max(a2_estimator.predict(next_state, env.actionSpace)) - a2_values(a2_action);
        end
        a1_estimator.update(state,a1_action,td_err1);
        a2_estimator.update(state,a2_action,td_err2);
        rwd(itr) = rwd(itr) + reward;
        state = next_state;
    end
    fprintf('final location: %d, %d; %d, %d\n',state');
    fprintf('%d th iteration, %d actions taken, final reward is %d, accumulated reward is %d.\n',itr,countActions,reward,rwd(itr));
    if isTraining
        iterationCount(itr) = countActions;
    end
end
if isTraining
    a1_weights = a1_estimator.weights;
    a2_weights = a2_estimator.weights;
    save('a1_weights.mat','a1_weights');
    save('a2_weights.mat','a2_weights');
    save('maq_iterationCount.mat','iterationCount');
    save('maq_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end