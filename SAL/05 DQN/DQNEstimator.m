classdef DQNEstimator < handle
	properties (SetAccess = private)
        env;
        alpha;
        
        weights;
        
        hidden_layer;
    end
	
	methods
		function obj = DQNEstimator(env,alpha,hidden_layer)
            obj.env = env;
            obj.alpha = alpha;
            
            obj.hidden_layer = hidden_layer;
            
            obj.weights.input = normrnd(0,1,[env.complexFeaturesLen+1, hidden_layer(1)])/sqrt(obj.env.complexFeaturesLen);
            obj.weights.hidden = normrnd(0,1,[hidden_layer(1)+1, hidden_layer(2)])/sqrt(hidden_layer(1));
			obj.weights.out = normrnd(0,1,[hidden_layer(2)+1, length(obj.env.actionSpace)])/sqrt(hidden_layer(2));
        end
        
        function set_weights(obj,weights)
            obj.weights = weights;
        end
        
		function value = predict(obj,state) 
            features = obj.env.get_complex_state_features(state);%features are already scaled.
            value.hidden_in_value = [1 features] * obj.weights.input;
            value.hidden_out_value = sigmoid(value.hidden_in_value);%activation function      
            
            value.hidden_in_value2 = [1 value.hidden_out_value] * obj.weights.hidden;
            value.hidden_out_value2 = sigmoid(value.hidden_in_value2);%activation function
            
            value.out_value = [1 value.hidden_out_value2] * obj.weights.out;
            
        end
        
        function update(obj,state,action,target)            
            features = [1 obj.env.get_complex_state_features(state)];
            value = obj.predict(state);
            out_value = value.out_value(action);
            hidden_out_value2 = value.hidden_out_value2;
            hidden_out_value = value.hidden_out_value;
            
            derivative_in(length(features), obj.hidden_layer(1)) = 0;
            for i=1:obj.hidden_layer(1)
                derivative_in(:,i) = (out_value - target) * ...
                                     sum(obj.weights.out(2:end,action)' .* ...
                                     (hidden_out_value2.*(1-hidden_out_value2)) .* ...
                                     obj.weights.hidden(i+1,:)) * ...
                                     hidden_out_value(i) * (1-hidden_out_value(i)) * features;
                obj.weights.input(:,i) = obj.weights.input(:,i) - obj.alpha * derivative_in(:,i);                    
            end
            derivative_hidden(obj.hidden_layer(2)+1, obj.hidden_layer(2)) = 0;
            for i=1:obj.hidden_layer(2)
                derivative_hidden(:,i) = (out_value - target) * obj.weights.out(i+1) * ...
                                          hidden_out_value2(i) * (1-hidden_out_value2(i)) * [1 hidden_out_value];
                obj.weights.hidden(:,i) = obj.weights.hidden(:,i) - obj.alpha * derivative_hidden(:,i);                    
            end
            
            derivative_out(:,1) = (out_value- target) * [1 hidden_out_value2];
            obj.weights.out(:,action) = obj.weights.out(:,action) - obj.alpha * derivative_out;
        end

    end
end