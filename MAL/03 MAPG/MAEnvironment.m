classdef MAEnvironment < handle
	properties (SetAccess = private)
        sizeMap = [30 30];%size of the map
		actionSpace = [1 2 3 4];
		locA = [7 7; 7 21];%default location of the agent
        locA_R = [7 21; 7 7];%default location of the agent
		locG = [14 14; 21 14];%location of the goal
		colorA = 'y';
		colorG = 'r';
        radii_A = 0.6;
		radii_G = 0.4;
		rwdG = 10;
		rwdC = -100; %collision
        
        current_location;
        simpleStateFeatureLen = 3; %for predict state values
        simpleFeatures = 10; %for predict state-action values
        simpleFeatureLen; % simpleFeatures * the length of action space
        
        complexFeaturesLen; %for complex feature abstraction

    end
	
	methods
		function obj = MAEnvironment()
			obj.current_location = obj.locA;
			obj.simpleFeatureLen = obj.simpleFeatures * length(obj.actionSpace); 
		end
		
		function reset(obj,location)
			if location > [0 0; 0 0] & location<= [obj.sizeMap; obj.sizeMap]
                obj.current_location = location;
            else
                if location == [0 0; 0 0] %random choose
                    obj.current_location = [ceil(rand(1)*obj.sizeMap(1)) ceil(rand(1)*obj.sizeMap(2));...
										  ceil(rand(1)*obj.sizeMap(1)) ceil(rand(1)*obj.sizeMap(2))];
                else
                    obj.current_location = obj.locA;
                end
            end
            change = false; %check if the random positions are the same or the same with goals
            if obj.current_location(1,:) == obj.locG(1,:) 
                change = true;
            end
            if obj.current_location(2,:) == obj.locG(1,:) 
                change = true;
            end
            if obj.current_location(1,:) == obj.locG(2,:) 
                change = true;
            end
            if obj.current_location(2,:) == obj.locG(2,:) 
                change = true;
            end
            if obj.current_location(1,:) == obj.current_location(2,:)
                change = true;
            end

            if change == true
                obj.current_location = obj.locA;
            end
        end
        
		function render(obj)
			axis([0 obj.sizeMap(1)+1 0 obj.sizeMap(2)+1]);
			cla;% Clear the axes.

			centers = obj.current_location(1,:);
			viscircles(centers,obj.radii_A,'Color',obj.colorA);
            txt =  strcat('\leftarrow Agnet', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2)+2*obj.radii_A,txt);
			
			centers = obj.current_location(2,:);
			viscircles(centers,obj.radii_A,'Color','b');
            txt =  strcat('\leftarrow Agnet', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2)+2*obj.radii_A,txt);

			centers = obj.locG(1,:);
			viscircles(centers,obj.radii_G,'Color',obj.colorG);
            txt =  strcat('\leftarrow Goal', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2),txt);
            
            centers = obj.locG(2,:);
			viscircles(centers,obj.radii_G,'Color',obj.colorG);
            txt =  strcat('\leftarrow Goal', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2),txt);
            
            %draw the wall
            hold on;
            x = [0, obj.sizeMap(1)+1, obj.sizeMap(1)+1, 0, 0];
            y = [0, 0, obj.sizeMap(2)+1, obj.sizeMap(2)+1, 0];
            plot(x, y, 'b', 'LineWidth', 2);
            
            pause(0.02);
		end
		
		function [next_state, reward, done] = step(obj,action) 
			done = false;
			reward = -1;
            old_state = obj.current_location;
			next_state = obj.current_location;
			
            switch action(1)
				case 1
					next_state(1,1) = next_state(1,1) + 1;
				case 2
					next_state(1,2) = next_state(1,2) + 1;
				case 3
					next_state(1,1) = next_state(1,1) - 1;
				case 4
					next_state(1,2) = next_state(1,2) - 1;
            end
			
			%to check if reached the wall
            if (min(next_state(1,:)) < 1) || next_state(1,1) > obj.sizeMap(1) || next_state(1,2) > obj.sizeMap(2) 
                next_state = obj.current_location;
            end
			obj.current_location = next_state;
			
            switch action(2)
				case 1
					next_state(2,1) = next_state(2,1) + 1;
				case 2
					next_state(2,2) = next_state(2,2) + 1;
				case 3
					next_state(2,1) = next_state(2,1) - 1;
				case 4
					next_state(2,2) = next_state(2,2) - 1;
            end
			%to check if reached the wall
            if (min(next_state(2,:)) < 1) || next_state(2,1) > obj.sizeMap(1) || next_state(2,2) > obj.sizeMap(2) 
                next_state = obj.current_location;
            end
			obj.current_location = next_state;
            
% 			%to check if reached the goal
% 			if isequal(next_state,obj.locG) || isequal([next_state(2,:) next_state(1,:)],obj.locG)
% 				reward(:) = obj.rwdG;
% 				done = true; %find the goal
%             end
            
            if (pdist([next_state(1,:); obj.locG(1,:)]) < 2 && ...
                    pdist([next_state(2,:); obj.locG(2,:)]) < 2) || ...
                    (pdist([next_state(1,:); obj.locG(2,:)]) < 2 && ...
                    pdist([next_state(2,:); obj.locG(1,:)]) < 2)
                reward = obj.rwdG;
				done = true; %find the goal
            end
            
			%to check if collide
            if isequal(next_state(1,:),next_state(2,:)) || isequal([next_state(2,:) next_state(1,:)],old_state)
				reward = obj.rwdC;
				done = true; %destroyed by the obstacle
            end
        end
        
        function features = get_scaled_simple_state_features(obj,state)
            max_dist = pdist([[1 1]; obj.sizeMap]);
            a1_to_goal1 = pdist([state(1,:); obj.locG(1,:)])/max_dist;
			a1_to_goal2 = pdist([state(1,:); obj.locG(2,:)])/max_dist;
			a2_to_goal1 = pdist([state(2,:); obj.locG(1,:)])/max_dist;
			a2_to_goal2 = pdist([state(2,:); obj.locG(2,:)])/max_dist;
			a1_to_a2 = pdist([state(1,:); state(2,:)])/max_dist;
            if a1_to_goal1+a2_to_goal2 > a1_to_goal2+a2_to_goal1
                a1_to_goal = a1_to_goal2;
                a2_to_goal = a2_to_goal1;
            else
                a1_to_goal = a1_to_goal1;
                a2_to_goal = a2_to_goal2;
            end
            features = [1 a1_to_goal a2_to_goal a1_to_a2];
        end
        
        function features = get_scaled_simple_features(obj,state,actions)
            max_angle = pi;
			a1_to_goal1 = clcAngle(obj.locG(1,:) - state(1,:));
			a1_to_goal2 = clcAngle(obj.locG(2,:) - state(1,:));
			a2_to_goal1 = clcAngle(obj.locG(1,:) - state(2,:));
			a2_to_goal2 = clcAngle(obj.locG(2,:) - state(2,:));
			a1_to_a2 = clcAngle(state(2,:) - state(1,:));
			
			sin_a1_goal1 = sin(a1_to_goal1)/max_angle;
            cos_a1_goal1 = cos(a1_to_goal1)/max_angle;
			sin_a1_goal2 = sin(a1_to_goal2)/max_angle;
			cos_a1_goal2 = cos(a1_to_goal2)/max_angle;
			sin_a2_goal1 = sin(a2_to_goal1)/max_angle;
            cos_a2_goal1 = cos(a2_to_goal1)/max_angle;
			sin_a2_goal2 = sin(a2_to_goal2)/max_angle;
			cos_a2_goal2 = cos(a2_to_goal2)/max_angle;
			sin_a1_a2 = sin(a1_to_a2)/max_angle;
			cos_a1_a2 = cos(a1_to_a2)/max_angle;

            feature = zeros(length(obj.actionSpace),obj.simpleFeatures);
            features(1:length(obj.actionSpace),1:obj.simpleFeatureLen + 1) = 0;
            for i=1:length(obj.actionSpace)
                feature(i,:) = [sin_a1_goal1 cos_a1_goal1 sin_a1_goal2 cos_a1_goal2 sin_a2_goal1...
								cos_a2_goal1 sin_a2_goal2 cos_a2_goal2 sin_a1_a2 cos_a1_a2];
                ftr = feature';
                features(i,:) = [1; ftr(:)]';
                feature(i,:) = 0;
            end
            features = features(actions,:);
        end

    end
end
