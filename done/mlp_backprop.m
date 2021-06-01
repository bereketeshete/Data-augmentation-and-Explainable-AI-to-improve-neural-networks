function [mlp, err_out] = mlp_backprop(mlp, err_in, type)
    % input : input data to current mlp layer
    % output : output feature of mlp layer
    
    if ~(strcmp(type, 'fc') || strcmp(type, 'relu'))
        error('Layer type should be either fc (fully connected) or relu (ReLU)');
    end
    
    %% Complete codes below
    if strcmp(type, 'fc')
        mlp.error=err_in;
        err_out=mlp.weight*err_in';
        %err_out = [];
    else
        err_out=(mlp.active.*(1-mlp.active)).*err_in';%
        %err_out = [];
    end
end