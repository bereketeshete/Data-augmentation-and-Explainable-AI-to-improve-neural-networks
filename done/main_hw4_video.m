% EE476_Audio_visual_perception_model
% Homework 4

%% STEP 0. Some Hyperparameters
clear all; clc; close all; rng(0);
student_id = 20160000;
your_name = 'your_name';
fprintf('[Homework 4-2 Video] name : %s, student id : %d \n', your_name, student_id )

opt.pooling_type = 'max'; % Pooling type, should be either 'max' or 'mean'
opt.hidden_size = 512; % Dimension of hidden layer of 2-layers MLP
opt.batch_size = 50; % Number of data in mini-batch
opt.init_std = 0.01; % Standard deviation of gaussian distribution where initial weight values are sampled
opt.learning_rate = 0.02; % Learning rate to be used in stochatic gradient decsent
opt.total_iteration = 50000; % Total number of mini-batch updates
opt.check_valid_freq = 1000; % How often you will check validation error
opt.print_train_freq = 100; % How often train error will be printed

%% STEP 1. Pre-processing raw data --- [DO NOT MODIFY]
data_dir = '../data/';
processed_dir = '../data/processed/';
model_dir = sprintf('trained_model_%s_%s_%s/', 'video', opt.pooling_type, datestr(now,'yymmddHHMMSS'));

if ~exist(processed_dir, 'dir')
    mkdir(processed_dir);
end
if ~exist(model_dir, 'dir')
    mkdir(model_dir);
end
if ~(exist(fullfile(processed_dir, 'video_processed.mat'), 'file') && ...
        exist(fullfile(processed_dir, 'data_label.mat'), 'file'))
    preprocess_data_label(data_dir, processed_dir);
end

load(fullfile(processed_dir, 'video_processed.mat'), 'video_processed');
train_video = video_processed{1};
valid_video = video_processed{2};
test_video = video_processed{3};
clear video_processed;

load(fullfile(processed_dir, 'data_label.mat'), 'label');
train_label = label{1};
valid_label = label{2};
test_label = label{3};
clear label;

fprintf('Video data and labels loaded \n');
%% STEP 2. Extract feature from pretrained CNNs --- [COMPLETE CODE FOR THIS STEP]
% Replaces the content of cell arrays
% 1) train_video 2) valid_video 3) test_video
% using CNNs you trained in Homework 3
% Structure of cell arrays and above should not be changed
% Data in each cell should have 
% (feature map height) X (feature map width) X (feature map number) X (number of time steps)













%% STEP 3. Extract feature from pretrained CNNs --- [DO NOT MODIFY]
% In this step we integrated features from each time step into one
% fixed-length feature
[train_video, valid_video, test_video] = global_pooling(train_video, valid_video, test_video, opt.pooling_type);

%% STEP 4-0. Training MLPs --- [DO NOT MODIFY]
% Initialize some parameters of 2-layers MLP
mlp(2) = struct();
relu(2) = struct();

mlp(1).input = [];
mlp(1).error = [];
mlp(1).weight = opt.init_std * randn(size(train_video, 2), opt.hidden_size);
mlp(1).bias = opt.init_std * randn(1, opt.hidden_size);
mlp(2).input = [];
mlp(2).error = [];
mlp(2).weight = opt.init_std * randn(opt.hidden_size, size(train_label, 2));
mlp(2).bias = opt.init_std * randn(1, size(train_label, 2));
relu(1).active = [];

% Weight update for every mini-batch
batch_start = 0;
batch_end = 0;
train_num = size(train_label, 1);
data_shuffle = randperm(train_num);
for i = 1:opt.total_iteration
    cur_batch = zeros(opt.batch_size, 1);
    batch_start = batch_end + 1;
    if batch_start > train_num
        batch_start = 1;
    end
    batch_end = batch_start + opt.batch_size - 1;
    if batch_end > train_num
        cur_batch(1:(train_num - batch_start + 1)) = data_shuffle(batch_start:train_num);
        data_shuffle = randperm(train_num);
        cur_batch((train_num - batch_start + 2):opt.batch_size) = data_shuffle(1:(batch_end - train_num));
        batch_end = batch_end - train_num;
    else
        cur_batch(:, 1) = data_shuffle(batch_start:batch_end);
    end
    
    %% STEP 4. 2-Layers MLP --- [COMPLETE CODE FOR THIS STEP]
    % Implement mlp_feedforward.m
    input = train_video(cur_batch, :);
    for j = 1:2
        [mlp(j), input] = mlp_feedforward(mlp(j), input, 'fc');
        if j == 1
            [relu(j), input] = mlp_feedforward(relu(j), input, 'relu');
        end
    end
    
    %% STEP 5. Softmax Regression --- [COMPLETE CODE FOR THIS STEP]
    % Implement softmax_regression.m
    [nll, ~, err] = softmax_regression(input, train_label(cur_batch, :));
    
    %% STEP 6-1. Error-back propagation --- [COMPLETE CODE FOR THIS STEP]
    % Implement mlp_backprop.m
    for j = fliplr(1:2)
        if j == 1
            [relu(j), err] = mlp_backprop(relu(j), err, 'relu');
        end
        [mlp(j), err] = mlp_backprop(mlp(j), err, 'fc');
    end
    
    %% STEP 6-2. Weight update using SGD --- [COMPLETE CODE FOR THIS STEP]
    % Implement weight_update.m
    for j = 1:2
        mlp(j) = weight_update(mlp(j), opt.learning_rate, opt.batch_size);
    end
    
    if rem(i, opt.print_train_freq) == 0
        fprintf('(%d / %d)  iteration, Error = %f\n', i, opt.total_iteration, nll)
    end
    
    if rem(i, opt.check_valid_freq) == 0
        [valid_nll, valid_acc] = evaluate(valid_video, valid_label, {mlp, relu});
        fprintf('Validation, (%d / %d)  iteration, Error = %f, Accuracy = %f%%\n', ...
            i, opt.total_iteration, valid_nll, 100 * valid_acc)
        
        mlp(1).input = [];
        mlp(1).error = [];
        mlp(2).input = [];
        mlp(2).error = [];
        relu(1).active = [];
        model = {mlp, relu};
        save(fullfile(model_dir, sprintf('model_%d_nll_%.2f_acc_%.2f.mat', i, valid_nll, 100*valid_acc)), 'model');
    end
end

%% EVALUATION ON TEST DATA WITH CHOSEN MODEL
% Use test code below to evaluate your best model with test data
%{
best_model = '';
load(best_model, 'model');
[test_nll, test_acc] = evaluate(test_video, test_label, model);

fprintf('Test, Error = %.2f, Accuracy = %.2f%%\n', ...
            test_nll, 100 * test_acc)
%}