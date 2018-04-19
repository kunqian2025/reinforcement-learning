function PE
close all;
clear; clc;

addpath('../Environment');
addpath('../Basic Functions');
env = SAEnvironment;

gamma = 1;% - discount factor
theta = 0.000001;% - max error for stopping iteration
Policy = ones(env.sizeMap(1),env.sizeMap(2),length(env.actionSpace))/length(env.actionSpace);
Policy(env.locG(1),env.locG(2),:) = 0;
Policy(env.locO(1),env.locO(2),:) = 0;
V = zeros(env.sizeMap(1),env.sizeMap(2));
% load('PE_V.mat','-mat');

V = policy_evaluation(env, Policy, V, gamma, theta);

save('PE_V.mat','V'); 



