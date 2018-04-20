function linear_function_approximation
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
isTesting = false; %declare if it is testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

alpha = 0.1; %learning rate settings 
gamma = 0.9; %discount factor
maxItr = 3000;%maximum iterations for ending one episode
estimator = LFAEstimator(env,alpha);

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    epsilon = 0.8; %random action choice
    min_epsilon = 0.3;
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    if isTesting
        NUM_ITERATIONS = 200;
        iterationCount(NUM_ITERATIONS) = 0;
        rwd(NUM_ITERATIONS) = 0;
    end
    epsilon = 0.3; %random action choice
    load('onp_lfa_weights.mat','-mat');    
    estimator.set_weights(Weights);
end


for itr=1:NUM_ITERATIONS 
    env.reset([0 0]);  
    if ~isTraining && ~isTesting 
		env.reset(env.locA);
        env.render();%display the moving environment
    end
    
    countActions = 0;%count how many actions in one iteration   
    reward = 0;
    done = false;
    
    agent_location = env.current_location;
    values = estimator.predict(agent_location, env.actionSpace);
    prob_a = make_epsilon_policy(values, epsilon);
    action = randsample(env.actionSpace,1,true,prob_a);
    val1 = values(action);
    
    while ~done   
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1;   
        
        [next_location, reward, done] = env.step(action);
        
        if ~isTraining
            if ~isTesting
                env.render();%display the moving environment
            end
            values = estimator.predict(next_location, env.actionSpace);
            prob_a = make_epsilon_policy(values, epsilon);
            next_action = randsample(env.actionSpace,1,true,prob_a);
            action = next_action;
            continue;
        end

        if ~done
            values = estimator.predict(next_location, env.actionSpace);
            prob_a = make_epsilon_policy(values, max(epsilon^log(itr),min_epsilon));
            next_action = randsample(env.actionSpace,1,true,prob_a);
            val2 = values(next_action);
        else
            val2 = 0;
        end
        td_err = reward + gamma*val2 - val1;
        estimator.update(agent_location,action,td_err);
        
        action = next_action;
        agent_location = next_location;
        val1 = estimator.predict(agent_location, action);

    end
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
%     fprintf('final location: %d, %d\n',env.current_location);
%     disp(estimator.weights);
    if isTraining  || isTesting
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
    end
end
if isTraining
    Weights = estimator.weights;
    save('onp_lfa_weights.mat','Weights');
    save('onp_lfa_iterationCount.mat','iterationCount');
    save('onp_lfa_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end
if isTesting
    figure,bar(iterationCount)
    meanA = num2str(mean(iterationCount));
    title(strcat('Mean steps: ',meanA));
    figure,bar(rwd)
end