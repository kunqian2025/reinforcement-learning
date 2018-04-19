function policy_gradient
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

p_estimator = PolicyEstimator(env,alpha);
v_estimator = ValueEstimator(env,alpha);

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    if isTesting
        NUM_ITERATIONS = 200;
        iterationCount(NUM_ITERATIONS) = 0;
        rwd(NUM_ITERATIONS) = 0;
    end
    load('policy_weights.mat','-mat');
    load('value_weights.mat','-mat');
    p_estimator.set_weights(policyWeights);
    v_estimator.set_weights(valueWeights);
end

for itr=1:NUM_ITERATIONS 
    env.reset(env.locA);
    if ~isTraining && ~isTesting 
        env.render();%display the moving environment
    end

    countActions = 0;%count how many actions in one iteration  
    reward = 0;
    done = false;
    state = env.current_location;
    fprintf('initial location: %d, %d\n',env.current_location);
    
    while ~done   
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1; 
        
        if ~isTraining
            prob_a = p_estimator.predict(state,env.actionSpace);
            action = randsample(env.actionSpace,1,true,prob_a);
            [next_state, reward, done] = env.step(action);
            state = next_state;
            if ~isTesting
                env.render();%display the moving environment
            end
            continue;
        end
        
        prob_a = p_estimator.predict(state,env.actionSpace);
        action = randsample(env.actionSpace,1,true,prob_a);
        
        [next_state, reward, done] = env.step(action);
                
        
        if ~done
            td_err = reward + gamma*v_estimator.predict(next_state) - v_estimator.predict(state);
        else
            td_err = reward - v_estimator.predict(state);
        end
        
        v_estimator.update(state,td_err);
        p_estimator.update(state,action,td_err);
        
        state = next_state;
    end
    fprintf('final location: %d, %d\n', state);
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
    if isTraining || isTesting
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
    end
end

if isTraining
    policyWeights = p_estimator.weights;
    valueWeights = v_estimator.weights;
    disp("policy weights:");
    disp(policyWeights);
    fprintf('value weights: %d, %d, %d\n',valueWeights);
    save('policy_weights.mat','policyWeights');
    save('value_weights.mat','valueWeights');
    save('pg_iterationCount.mat','iterationCount');
    save('pg_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end
if isTesting
    figure,bar(iterationCount)
    meanA = num2str(mean(iterationCount));
    title(strcat('Mean steps: ',meanA));
    figure,bar(rwd)
end

