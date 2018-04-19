classdef SAEnvironment < handle
	properties (SetAccess = private)
%         sizeMap = [20 20];%size of the map
% 		actionSpace = [1 2 3 4];
%         actionPossibilities = [0.8 0.1 0.1];
% 		locA = [5 5];%default location of the agent
% 		locG = [15 15];%location of the goal
% 		locO = [10 10];%location of the obstacle
		sizeMap = [30 30];%size of the map
		actionSpace = [1 2 3 4];
        actionPossibilities = [0.8 0.1 0.1];
		locA = [5 5];%default location of the agent
		locG = [25 25];%location of the goal
		locO = [15 15];%location of the obstacle
		colorA = 'y';
		colorG = 'r';
		colorO = 'g';
        radii_A = 0.6;
		radii_G = 0.4;
        radii_O = 0.4;
		rwdG = 1;
		rwdO = -1;
        
        current_location;
        simpleStateFeatureLen = 2; %for predict state values
        simpleFeatures = 4; %for predict state-action values
        simpleFeatureLen; % simpleFeatures * the length of action space
        
        complexFeaturesLen; %for complex feature abstraction

    end
	
	methods
		function obj = SAEnvironment()
			obj.current_location = obj.locA;
			obj.simpleFeatureLen = obj.simpleFeatures * length(obj.actionSpace); 
			obj.complexFeaturesLen = obj.sizeMap(1)*obj.sizeMap(2)*3;
		end
		
		function reset(obj,location)
            if location > [0 0] & location<= obj.sizeMap
                obj.current_location = location;
            else
                if location == [0 0] %random choose
                    obj.current_location = [ceil(rand(1)*obj.sizeMap(1)) ceil(rand(1)*obj.sizeMap(1))];
                else
                    obj.current_location = obj.locA;
                end
            end
            if obj.current_location == obj.locG 
				obj.current_location = obj.locA;
			else 
				if obj.current_location == obj.locO
					obj.current_location = obj.locA;
				end
            end
        end
        
		function render(obj)
			axis([0 obj.sizeMap(1)+1 0 obj.sizeMap(2)+1]);
			cla;% Clear the axes.

			centers = obj.current_location;
			viscircles(centers,obj.radii_A,'Color',obj.colorA);
            txt =  strcat('\leftarrow Agnet', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2)+2*obj.radii_A,txt);

			centers = obj.locG;
			viscircles(centers,obj.radii_G,'Color',obj.colorG);
            txt =  strcat('\leftarrow Goal', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2),txt);

			centers = obj.locO;
			viscircles(centers,obj.radii_O,'Color',obj.colorO);
            txt =  strcat('\leftarrow Obstacle', '(', num2str(centers(1)), ',',  num2str(centers(2)), ')');
            text(centers(1),centers(2),txt);
            
            %draw the wall
            hold on;
            x = [0, obj.sizeMap(1)+1, obj.sizeMap(1)+1, 0, 0];
            y = [0, 0, obj.sizeMap(2)+1, obj.sizeMap(2)+1, 0];
            plot(x, y, 'b', 'LineWidth', 2);
            
            pause(0.02);
		end
		
		function [prob, next_state, reward] = possible_next_state(obj, state, action)
			prob = obj.actionPossibilities;
			reward = [0 0 0];
            switch action 
				case 1
                    actions = [1 2 4];
				case 2
					actions = [2 3 1];
				case 3
					actions = [3 4 2];
				case 4
					actions = [4 1 3];
                otherwise
					disp('wrong action input, will not move');
            end
			next_state(1:length(actions),1:2) = 0;
			for action=1:length(actions)
				next_state(action,:) = state;
				switch actions(action)
					case 1
						next_state(action,1) = next_state(action,1) + 1;
					case 2
						next_state(action,2) = next_state(action,2) + 1;
					case 3
						next_state(action,1) = next_state(action,1) - 1;
					case 4
						next_state(action,2) = next_state(action,2) - 1;
				end
				%to check if reached the wall
				if (min(next_state(action,:)) < 1) || next_state(action,1) > obj.sizeMap(1) || next_state(action,2) > obj.sizeMap(2) 
					next_state(action,:) = state;
				end
				%to check if reached the goal
				if next_state(action,:) == obj.locG %pdist([next_state; obj.locG]) < 1
					reward(action) = obj.rwdG;
				end
				%to check if reached the obstacle
				if next_state(action,:) == obj.locO %pdist([next_state; obj.locO]) < 1
					reward(action) = obj.rwdO;
				end
			end
		end
		
		function [next_state, reward, done] = step(obj,action) 
			done = false;
			reward = 0;
			next_state = obj.current_location;
            
            switch action %for non-deterministic environment
				case 1
                    action = randsample([1 2 4],1,true,obj.actionPossibilities);
				case 2
					action = randsample([2 3 1],1,true,obj.actionPossibilities);
				case 3
					action = randsample([3 4 2],1,true,obj.actionPossibilities);
				case 4
					action = randsample([4 1 3],1,true,obj.actionPossibilities);
                otherwise
					disp('wrong action input, will not move');
            end
            
            switch action
				case 1
					next_state(1) = next_state(1) + 1;
				case 2
					next_state(2) = next_state(2) + 1;
				case 3
					next_state(1) = next_state(1) - 1;
				case 4
					next_state(2) = next_state(2) - 1;
            end
			%to check if reached the wall
            if (min(next_state) < 1) || next_state(1) > obj.sizeMap(1) || next_state(2) > obj.sizeMap(2) 
                next_state = obj.current_location;
            end
            obj.current_location = next_state;
            
			%to check if reached the goal
			if next_state == obj.locG %pdist([next_state; obj.locG]) < 1
				reward = obj.rwdG;
				done = true; %find the goal
			end
			%to check if reached the obstacle
			if next_state == obj.locO %pdist([next_state; obj.locO]) < 1
				reward = obj.rwdO;
				done = true; %destroyed by the obstacle
			end
        end
        
        function features = get_scaled_simple_state_features(obj,state)
            max_dist = pdist([[1 1]; obj.sizeMap]);
            dist_to_goal = pdist([state; obj.locG])/max_dist;
            dist_to_obstacle = pdist([state; obj.locO])/max_dist;
            features = [1 dist_to_goal dist_to_obstacle];
        end
        
        function features = get_scaled_simple_features(obj,state,actions)
            max_angle = pi;
            angle_to_goal = clcAngle(obj.locG - state);
            angle_to_obstacle = clcAngle(obj.locO - state);
            sin_goal = sin(angle_to_goal)/max_angle;
            cos_goal = cos(angle_to_goal)/max_angle;
            sin_obstacle = sin(angle_to_obstacle)/max_angle;
            cos_obstacle = cos(angle_to_obstacle)/max_angle;
            feature = zeros(length(obj.actionSpace),obj.simpleFeatures);
            features(1:length(obj.actionSpace),1:obj.simpleFeatureLen + 1) = 0;
            for i=1:length(obj.actionSpace)
                feature(i,:) = [sin_goal cos_goal sin_obstacle cos_obstacle];
                ftr = feature';
                features(i,:) = [1; ftr(:)]';
                feature(i,:) = 0;
            end
            features = features(actions,:);
        end
		
		function features = get_complex_state_features(obj,state)    	
            matrix_value_a = zeros(obj.sizeMap);
            matrix_value_a(state(1),state(2)) = 1;
            matrix_value_g = zeros(obj.sizeMap);
            matrix_value_g(obj.locG(1),obj.locG(2)) = 1;
            matrix_value_o = zeros(obj.sizeMap);
            matrix_value_o(obj.locO(1),obj.locO(2)) = 1;
            features = [matrix_value_a(:)' matrix_value_g(:)' matrix_value_o(:)'];
        end
		
    end
end
