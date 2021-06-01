function [train_out, valid_out, test_out] = global_pooling(train, valid, test, pool_type)
    if ~(strcmp(pool_type, 'mean') || strcmp(pool_type, 'max'))
        error('Pooling type should be either mean or max');
    end

    time_dim = ndims(train{1,1});
    out_dim = numel(train{1,1}) / size(train{1,1}, time_dim);

    train_out = zeros(size(train, 1), out_dim);
    valid_out = zeros(size(valid, 1), out_dim);
    test_out = zeros(size(test, 1), out_dim);

    for i = 1:size(train, 1)
        if strcmp(pool_type, 'max')
            train_out(i, :) = reshape(max(train{i, 1}, [], time_dim), [out_dim, 1]);
        else
            train_out(i, :) = reshape(mean(train{i, 1}, time_dim), [out_dim, 1]);
        end
    end

    for i = 1:size(valid, 1)
        if strcmp(pool_type, 'max')
            valid_out(i, :) = reshape(max(valid{i, 1}, [], time_dim), [out_dim, 1]);
        else
            valid_out(i, :) = reshape(mean(valid{i, 1}, time_dim), [out_dim, 1]);
        end
    end

    for i = 1:size(test, 1)
        if strcmp(pool_type, 'max')
            test_out(i, :) = reshape(max(test{i, 1}, [], time_dim), [out_dim, 1]);
        else
            test_out(i, :) = reshape(mean(test{i, 1}, time_dim), [out_dim, 1]);
        end
    end

end