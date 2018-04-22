classdef ValueEstimator < handle
	properties (SetAccess = private)
        env;
        alpha;
        weights;
    end
	
	methods
		function obj = ValueEstimator(env,alpha)
            obj.env = env;
            obj.alpha = alpha;
            
            obj.weights = rand(env.simpleStateFeatureLen + 1,1);
        end
        
        function set_weights(obj,weights)
            obj.weights = weights;
        end
        
		function value = predict(obj,state) 
            features = obj.env.get_scaled_simple_state_features(state);%features are already scaled.
            value = features * obj.weights;
        end
        
        function update(obj,state,td_err)
            features = obj.env.get_scaled_simple_state_features(state);
            obj.weights = obj.weights + obj.alpha * td_err * features';
        end

    end
end