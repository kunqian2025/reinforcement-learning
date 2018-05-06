function compare_fig(rwd)
%% display orignal value
figure;
hold on;
plot(rwd(1,:),'r.','MarkerSize',0.5);
plot(rwd(2,:),'g.','MarkerSize',0.5);
plot(rwd(3,:),'b.','MarkerSize',0.5);
xlabel('Episodes'); % x-axis label
ylabel('Sum of rewards during episode'); % y-axis label

legend('Multi-Agent Actor-Critic','Hysteretic Q-Learning','Centralized Q-Learning');
hold off;

%% display statistical features
figure;
hold on;
xlabel('Episodes'); % x-axis label
ylabel('Sum of rewards during episode'); % y-axis label
step = 200;
value_len = length(rwd(1,:));
itr = ceil(value_len/step);
mean_value(itr) = 0;
std_d(itr) = 0;
for i = 1:itr-1
    y = hampel(rwd(1,(i-1)*step+1:i*step));
    mean_value(i) = mean(y);
    std_d(i) = std(y);
end
y = hampel(rwd(1,(itr-1)*step+1:value_len));
mean_value(itr) = mean(y);
std_d(itr) = std(y);
x = step/2:step:value_len;
errorbar(x,mean_value,std_d,'-s','MarkerSize',4,...
    'MarkerEdgeColor','red','LineWidth', 1,'Color','red');
for i = 1:itr-1
    y = hampel(rwd(2,(i-1)*step+1:i*step));
    mean_value(i) = mean(y);
    std_d(i) = std(y);
end
y = hampel(rwd(2,(itr-1)*step+1:value_len));
mean_value(itr) = mean(y);
std_d(itr) = std(y);
x = step/2:step:value_len;
errorbar(x,mean_value,std_d,'-s','MarkerSize',4,...
    'MarkerEdgeColor','green','LineWidth', 1,'Color','green');
for i = 1:itr-1
    y = hampel(rwd(3,(i-1)*step+1:i*step));
    mean_value(i) = mean(y);
    std_d(i) = std(y);
end
y = hampel(rwd(3,(itr-1)*step+1:value_len));
mean_value(itr) = mean(y);
std_d(itr) = std(y);
x = step/2:step:value_len;
errorbar(x,mean_value,std_d,'-s','MarkerSize',4,...
    'MarkerEdgeColor','blue','LineWidth', 1,'Color','blue');

legend('Multi-Agent Actor-Critic','Hysteretic Q-Learning','Centralized Q-Learning');
hold off;