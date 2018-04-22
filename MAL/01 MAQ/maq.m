function maq
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Basic Functions');
env = MAEnvironment;

alpha = 0.1; %learning rate settings 
beta = 0.01;
gamma = 0.99; %discount factor
maxItr = 6000;%maximum iterations for ending one episode
a1_estimator = LFAEstimator(env,alpha,beta);
a2_estimator = LFAEstimator(env,alpha,beta);

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    epsilon = 0.8; %random action choice
    min_epsilon = 0.3;
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    epsilon = 0.3; %random action choice
    load('a1_weights.mat','-mat');
    load('a2_weights.mat','-mat');
    a1_estimator.set_weights(a1_weights);
    a2_estimator.set_weights(a2_weights);
end

for itr=1:NUM_ITERATIONS 
    env.reset([0 0; 0 0]); 
    if ~isTraining
        env.reset(env.locA_R); 
        env.render();%display the moving environment
    end
	state = env.current_location;
    fprintf('initial location: %d, %d; %d, %d\n',state');
    
    countActions = 0;%count how many actions in one iteration  
    reward = 0;
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
            state = next_state;
            env.render();%display the moving environment
            continue;
        end
        a1_values = a1_estimator.predict(state, env.actionSpace);
        prob_a = make_epsilon_policy(a1_values, max(epsilon^log(itr),min_epsilon));
        a1_action = randsample(env.actionSpace,1,true,prob_a);
        
        a2_values = a2_estimator.predict(state, env.actionSpace);
        prob_a = make_epsilon_policy(a2_values, max(epsilon^log(itr),min_epsilon));
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
        state = next_state;
    end
    fprintf('final location: %d, %d; %d, %d\n',state');
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
    if isTraining
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
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