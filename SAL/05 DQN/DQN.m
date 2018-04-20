function DQN
close all;
clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isTraining = false; %declare if it is training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

alpha = 0.1; %learning rate settings 
gamma = 0.9; %discount factor
maxItr = 3000;%maximum iterations for ending one episode

hidden_layer = [40 40];
estimator = DQNEstimator(env,alpha,hidden_layer);

if isTraining
    %replay buffer
    memory_size = 30000;
    memory_cnt = 0;
    batch_size = 3000;
    memory_buffer(1:memory_size) = struct('state',[],'action',[],'next_state',[],'reward',[],'done',[]);
    
    
    NUM_ITERATIONS = 20000; %change this value to set max iterations
    epsilon = 0.8; %random action choice
    min_epsilon = 0.3;
    iterationCount(NUM_ITERATIONS) = 0;
    rwd(NUM_ITERATIONS) = 0;
else
    NUM_ITERATIONS = 5;
    epsilon = 0.3; %random action choice
    load('DQN_weights.mat','-mat');
    estimator.set_weights(Weights);
end
timeStart = clock;
for itr=1:NUM_ITERATIONS 
    env.reset([0 0]);  
    if ~isTraining
        env.reset(env.locA);  
        env.render();%display the moving environment
    end
    
    countActions = 0;%count how many actions in one iteration  
    reward = 0;
    done = false;
    state = env.current_location;
    
    while ~done   
        if countActions == maxItr
            break;
        end
        countActions = countActions + 1; 
        
        if ~isTraining
            values = estimator.predict(state).out_value;
            prob_a = make_epsilon_policy(values, epsilon);
            action = randsample(env.actionSpace,1,true,prob_a);      
        
            [next_state, reward, done] = env.step(action);
            
            state = next_state;
            env.render();%display the moving environment
            continue;
        end

        values = estimator.predict(state).out_value;
        prob_a = make_epsilon_policy(values, max(epsilon^log(itr),min_epsilon));
        action = randsample(env.actionSpace,1,true,prob_a);      
        
        [next_state, reward, done] = env.step(action);

%         target = reward;
%         if ~done
%                 target = reward + gamma*max(estimator.predict(next_state).out_value);
%         end
%         estimator.update(state,action,target);
        memory_buffer(2:memory_size) = memory_buffer(1:memory_size-1);
        memory_buffer(1).state = state;
        memory_buffer(1).action = action;
        memory_buffer(1).next_state = next_state;
        memory_buffer(1).reward = reward;
        memory_buffer(1).done = done;
        memory_cnt = memory_cnt + 1; 
        
        state = next_state;
    end
    fprintf('%d th iteration, %d actions taken, final reward is %d.\n',itr,countActions,reward);
    if isTraining
        iterationCount(itr) = countActions;
        rwd(itr) = reward;
        %memory replay
        if memory_cnt >= memory_size
            mini_batch = randsample(memory_buffer,batch_size);
            for i=1:batch_size
                tem_state = mini_batch(i).state;
                tem_action = mini_batch(i).action;
                tem_next_state = mini_batch(i).next_state;
                tem_reward = mini_batch(i).reward;
                tem_done = mini_batch(i).done;
                tem_next_state_values = estimator.predict(tem_next_state).out_value;
                tem_target = tem_reward;
                if ~tem_done
                    tem_target = tem_reward + gamma*max(tem_next_state_values);
                end
                estimator.update(tem_state,tem_action,tem_target);
            end
        end
    end
end
if isTraining
    timeEnd = clock;
    timeDiff = sum([timeEnd - timeStart].*[0 0 0 3600 60 1]);
    simulationTime = [timeStart timeEnd timeDiff];
    save('DQN_simulationTime.mat','simulationTime'); 
    Weights = estimator.weights;
    save('DQN_weights.mat','Weights');
    save('DQN_iterationCount.mat','iterationCount');
    save('DQN_reward.mat','rwd');
    figure,bar(iterationCount)
    figure,bar(rwd)
end


