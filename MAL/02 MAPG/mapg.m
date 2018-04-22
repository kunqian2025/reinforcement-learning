function mapg
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Basic Functions');
env = MAEnvironment;

alpha = 0.005; %learning rate settings 
gamma = 0.99; %discount factor
maxItr = 6000;%maximum iterations for ending one episode

a1_p_estimator = PolicyEstimator(env,alpha);
a2_p_estimator = PolicyEstimator(env,alpha);
v_estimator = ValueEstimator(env,alpha*10);

if isTraining
    NUM_ITERATIONS = 100000; %change this value to set max iterations
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    load('agent1_policy_weights.mat','-mat');
    load('agent2_policy_weights.mat','-mat');
    load('value_weights.mat','-mat');
    a1_p_estimator.set_weights(a1_p_weights);
    a2_p_estimator.set_weights(a2_p_weights);
    v_estimator.set_weights(v_weights);
end

for itr=1:NUM_ITERATIONS 
    env.reset([0 0; 0 0]); 
    if ~isTraining
        env.reset(env.locA); 
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
            prob_a1 = a1_p_estimator.predict(state,env.actionSpace);
            prob_a2 = a2_p_estimator.predict(state,env.actionSpace);
            action = [randsample(env.actionSpace,1,true,prob_a1) ...
                      randsample(env.actionSpace,1,true,prob_a2)]; 
%             action = [action(1) 0];
            [next_state, reward, done] = env.step(action);
            state = next_state;
            env.render();%display the moving environment
            continue;
        end
        
        prob_a1 = a1_p_estimator.predict(state,env.actionSpace);
        prob_a2 = a2_p_estimator.predict(state,env.actionSpace);
        action = [randsample(env.actionSpace,1,true,prob_a1) ...
                  randsample(env.actionSpace,1,true,prob_a2)];  

        [next_state, reward, done] = env.step(action);
        
        if done
            td_err = reward - v_estimator.predict(state);
        else
            td_err = reward + gamma*v_estimator.predict(next_state) - v_estimator.predict(state);
        end
        v_estimator.update(state,td_err);
        a1_p_estimator.update(state,action(1),td_err);
        a2_p_estimator.update(state,action(2),td_err);
        
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
    a1_p_weights = a1_p_estimator.weights;
    a2_p_weights = a2_p_estimator.weights;
    v_weights = v_estimator.weights;
    save('agent1_policy_weights.mat','a1_p_weights');
    save('agent2_policy_weights.mat','a2_p_weights');
    save('value_weights.mat','v_weights');
    save('mapg_iterationCount.mat','iterationCount');
    save('mapg_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end