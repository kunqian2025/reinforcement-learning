classdef LFAEstimator < handle
	properties (SetAccess = private)
        env;
        weights;
        alpha;
        beta;
    end
	
	methods
		function obj = LFAEstimator(env,alpha,beta)
            obj.env = env;
            obj.weights = rand(1, env.simpleFeatureLen+1);
            obj.alpha = alpha;
            obj.beta = beta;
        end
        
        function set_weights(obj,weights)
            obj.weights = weights;
        end
        
		function value = predict(obj,state,actions)
            features = obj.env.get_scaled_simple_features(state,actions);
            value = obj.weights*features';
        end
        
        function update(obj,state,action,td_err)
            features = obj.env.get_scaled_simple_features(state,action);
            if td_err>0
                obj.weights = obj.weights + obj.alpha * td_err * features;
            else
                obj.weights = obj.weights + obj.beta * td_err * features;
            end
            
        end

    end
end