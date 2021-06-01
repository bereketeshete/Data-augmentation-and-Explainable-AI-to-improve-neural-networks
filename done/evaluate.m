function [nll, acc] = evaluate(input, label, layers)
    mlp = layers{1};
    relu = layers{2};
    data_num = size(input, 1);
    
    total_nll = 0;
    acc = 0;
    for i = 1:data_num
        output = input(i, :);
        for j = 1:2
            [~, output] = mlp_feedforward(mlp(j), output, 'fc');
            if j == 1
                [~, output] = mlp_feedforward(relu(j), output, 'relu');
            end
        end
        [nll, softmax, ~] = softmax_regression(output, label(i, :));
        %softmax
        %disp('check me');
        total_nll = total_nll + nll;
        [~, predict] = max(softmax);
        [~, true_label] = max(label(i, :));
        if predict == true_label
            %disp('hop');
            acc = acc + 1;
        end
    end
    
    nll = total_nll / data_num;
    acc = acc / data_num;
end