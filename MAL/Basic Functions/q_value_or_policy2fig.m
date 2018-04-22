function q_value_or_policy2fig(q_value_or_policy,goal_state,obstacle_state)

values = q_value_or_policy;
numGoal = length(goal_state(:,1));
numObstacle = length(obstacle_state(:,1));

xLength = length(values(:,1,1));
yLength = length(values(1,:,1));

totalHeight = xLength;
totalWidth = yLength;
figure;
axis([0 totalHeight 0 totalWidth]);
axis off;
hold on;

triangleRowBase = totalHeight/xLength;
triangleColumnBase = totalWidth/yLength;

for kk = 1:yLength
    for jj = 1:xLength
        top = triangleRowBase*(kk);
        bottom = triangleRowBase*(kk-1);
        left = triangleColumnBase*(jj-1);
        right = triangleColumnBase*(jj);
        
        filled = false;
        for ii = 1:numGoal %fill the goal with red
            if [jj, kk] == goal_state(ii,:)
                fill([right right left left right], [top bottom bottom top top], 'r');
                filled = true;
                break;
            end
        end
        if filled
            continue;
        end
        for ii = 1:numObstacle %fill the obstacle with green
            if [jj, kk] == obstacle_state(ii,:)
                fill([right right left left right], [top bottom bottom top top], 'g');
                filled = true;
                break;
            end
        end
        if filled
            continue;
        end
        
        centerX = (left+right)/2;
        centerY = (top+bottom)/2;
        
        color = ['w','w','w','w'];
        max_value = max(values(jj,kk,:));
        for ii=1:4
            if values(jj,kk,ii) == max_value
                color(ii) = 'b';
            end
        end
        
        fill([right right centerX right], [top bottom centerY top],color(1));
        fill([right left centerX right], [top top centerY top], color(2));           
        fill([left left centerX left], [top bottom centerY top], color(3));          
        fill([left right centerX left], [bottom bottom centerY bottom], color(4));
         
%         [xS1,yS1] = ds2nfu(centerX+0.5*(centerX-left),centerY);
%         dim1 = [xS1 yS1 0 0];       
%         [xS2,yS2] = ds2nfu(centerX,centerY+0.5*(top-centerY));
%         dim2 = [xS2 yS2 0 0];       
%         [xS3,yS3] = ds2nfu(centerX-0.5*(centerX-left),centerY);
%         dim3 = [xS3 yS3 0 0];       
%         [xS4,yS4] = ds2nfu(centerX,centerY-0.5*(top-centerY));
%         dim4 = [xS4 yS4 0 0];       
% 
%         str1 = sprintf('%.1f', values(jj,kk,1));
%         str2 = sprintf('%.1f', values(jj,kk,2));
%         str3 = sprintf('%.1f', values(jj,kk,3));
%         str4 = sprintf('%.1f', values(jj,kk,4));
%         annotation('textbox',dim1,'String',str1,'FitBoxToText','on', 'HorizontalAlignment', 'center', 'VerticalAlignment','middle', 'LineStyle','none');
%         annotation('textbox',dim2,'String',str2,'FitBoxToText','on', 'HorizontalAlignment', 'center', 'VerticalAlignment','middle', 'LineStyle','none');
%         annotation('textbox',dim3,'String',str3,'FitBoxToText','on', 'HorizontalAlignment', 'center', 'VerticalAlignment','middle', 'LineStyle','none');
%         annotation('textbox',dim4,'String',str4,'FitBoxToText','on', 'HorizontalAlignment', 'center', 'VerticalAlignment','middle', 'LineStyle','none');
    end
end