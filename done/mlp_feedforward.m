function [mlp, output] = mlp_feedforward(mlp, input, type)
    % input : input data to current mlp layer
    % output : output feature of mlp layer
    % mlp.weight : concatenation of weight and bias
    
    if ~(strcmp(type, 'fc') || strcmp(type, 'relu'))
        error('Layer type should be either fc (fully connected) or relu (ReLU)');
    end
    
    %% Complete codes below
    if strcmp(type, 'fc') 
        mlp.input=input;
        if size(input,1)~=size(mlp.bias,1)
            repmat(mlp.bias,[size(input,1) 1]);
        end 
        output = input * mlp.weight + mlp.bias;
        %output = [];
    else
        mlp.active=max(0,input);
        output=mlp.active;
        %output = [];
    end
end

