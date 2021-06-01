function mlp = weight_update(mlp, learning_rate, batch_size)
    % Complete codes below
    %weight_gradient = [];
    
    weight_gradient=-(learning_rate*mlp.input'*mlp.error)/batch_size;
    %bias_gradient = [];
    bias_gradient=-learning_rate*mlp.error/batch_size;
    %mlp.weight = [];
    mlp.weight=mlp.weight+weight_gradient;
    %mlp.bias = [];
    mlp.bias= mlp.bias;%+bias_gradient;%-(learning_rate*mlp.input'*mlp.error)/batch_size
end