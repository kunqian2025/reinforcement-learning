classdef PolicyEstimator < handle
	properties (SetAccess = private)
        env;
        alpha;
        weights;
    end
	
	methods
		function obj = PolicyEstimator(env,alpha)
            obj.env = env;
            obj.alpha = alpha;
            obj.weights = rand(env.simpleFeatureLen + 1,1);
        end
        
        function set_weights(obj,weights)
            obj.weights = weights;
        end
        
		function probabilities = predict(obj,state,actions) 
            features = obj.env.get_scaled_simple_features(state,actions);%features are already scaled.
            preference = features * obj.weights;
            probabilities = exp(preference)/sum(exp(preference));
        end
        
        function update(obj,state,action,td_err)
            features = obj.env.get_scaled_simple_features(state,obj.env.actionSpace);
            probabilities = predict(obj,state,obj.env.actionSpace);
            derivative = features(action,:) - sum(probabilities.*features);%for linear approximation with softmax
            obj.weights = obj.weights + obj.alpha * td_err * derivative';
        end

    end
end