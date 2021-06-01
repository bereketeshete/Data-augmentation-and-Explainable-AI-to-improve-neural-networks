function [nll, softmax, err] = softmax_regression(input, label)
    % input : input data to current mlp layer with size (batch_size) X (feature_dim)
    % label : label of input data (batch_size) X 10 
    % nll : cross-entropy error
    % err : error to be back-propagated to 2-layer mlp
    
    %% Complete codes below
    % softmax function output
    temp=exp(input);
    softmax= temp./sum(temp,2);
    softmax = double(bsxfun(@eq, softmax, max(softmax, [], 2)));
    %softmax = [];
    
    % Calculate cross entropy error using output from softmax
    y_hat=sum(softmax);
    y=sum(label);%label
    nll=sum(sum(-y.*log10(y_hat)-(1-y).*log10(1-y_hat)))/size(label,1);% cross entropy loss (error)
    %nll = [];
    %nll=100;
    % Calculate error 
    err = softmax - label;
    %err = 0;
end