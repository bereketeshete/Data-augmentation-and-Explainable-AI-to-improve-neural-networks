% EE476_Audio_visual_perception_model
% Homework 4
% Term Project :Bereket Eshete
%% STEP 0. Some Hyperparameters
clc; close all; rng(0);clear all; 
student_id = 20150923;
your_name = 'your_name';
fprintf('[Homework 4-1 Audio] name : %s, student id : %d \n', your_name, student_id )
main_task1
c=train_audio; % layers from CNN

opt.pooling_type = 'max'; %max Pooling type, should be either 'max' or 'mean'
opt.hidden_size = 96; % 96 Dimension of hidden layer of 2-layers MLP
opt.batch_size = 100; % Number of data in mini-batch
opt.init_std = 0.01; % 0.01Standard deviation of gaussian distribution where initial weight values are sampled
opt.learning_rate = 0.02; % 0.02 Learning rate to be used in stochatic gradient decsent
opt.total_iteration = 50000; %50000 Total number of mini-batch updates
opt.check_valid_freq = 1000; % 1000 How often you will check validation error
opt.print_train_freq = 100; % How often train error will be printed

%% STEP 1. Pre-processing raw data --- [DO NOT MODIFY]
data_dir = '../data/train';
processed_dir = '../data/processed/';
test_dir='../data/test/audio';
model_dir = sprintf('trained_model_%s_%s_%s/', 'audio', opt.pooling_type, datestr(now,'yymmddHHMMSS'));


if ~exist(processed_dir, 'dir')
    mkdir(processed_dir);
end
if ~exist(model_dir, 'dir')
    mkdir(model_dir);
end
if ~(exist(fullfile(processed_dir, 'audio_processed.mat'), 'file') && ...
        exist(fullfile(processed_dir, 'data_label.mat'), 'file'))
    preprocess_data_label(data_dir, processed_dir);
end

load(fullfile(processed_dir, 'audio_processed.mat'), 'audio_logmel');
train_audio = audio_logmel{1};
valid_audio = audio_logmel{2};
test_audio = audio_logmel{3};
clear audio_logmel;


load(fullfile(processed_dir, 'data_label.mat'), 'label');
train_label = label{1};
valid_label = label{2};
test_label = label{3};
clear label;

fprintf('Audio data and labels loaded \n');
%% STEP 2. Extract feature from pretrained CNNs --- [COMPLETE CODE FOR THIS STEP]
% Replaces the content of cell arrays
% 1) train_audio 2) valid_audio 3) test_audio
% using CNNs you trained in Homework 3
% Structure of cell arrays and above should not be changed
% Data in each cell should have (feature dimension) X (number of time steps)
% weight 1 [26x96],weight 2 [96x10],bias 1[1x96] ,bias 2[1x10]

% train_audio=train_audio(1:800,:);
% train_label=train_label(1:800,:);

new_idea='none';% 'only_mlp' or 'none'
cnn_output=c(:,1);
if ~strcmp(new_idea,'only_mlp')
    cnn_output=c(:,3);
    for i=1:900
     b{i}=squeeze(cnn_output{i})';
    end
    cnn_output=b';
end 
valid_audio=cnn_output(851:900,:); % chosse for test or train data
train_audio=cnn_output(1:850,:);
test_audio=cnn_output(1:150,:);


%%  Step Additional. Data augmentation  --- [Term Project]

% specify data_augmentation method
method='flip';  % none,noise,resize,repeat,flip
if strcmp(method,'noise')
      
      for i=1:size(train_audio,1)
          m=train_audio(i);m=m{1};
          m=imnoise(m,'poisson');% specify the type of noise(gaussian,slat & pepper,speckle,poisson)
          data_augment{i,1}=m;
      end
      train_audio=cat(1,train_audio,data_augment);% add augmented to the training data
      train_label=repmat(train_label,[2 1]);
      
elseif strcmp(method,'repeat')
    disp('rep');
    rep=2;% number of times to repeat training data
    data_augment=train_audio; 
    train_audio=repmat(train_audio,[rep 1]);
    train_label=repmat(train_label,[rep 1]);
    
elseif strcmp(method,'flip')
    for i=1:size(train_audio,1)
          m=train_audio(i);m=m{1};
          m_hor=flip(m,2);
          m_ver=flip(m,1);
          data_augment_1{i,1}=m_hor;
          data_augment_2{i,1}=m_ver;
    end
      train_audio=cat(1,train_audio,data_augment_1,data_augment_2);% add augmented to the training data
      train_label=repmat(train_label,[3 1]);
      
elseif strcmp(method,'resize')
    for i=1:size(train_audio,1)
          m=train_audio(i);m=m{1};
          m_half=imresize(m,[size(m,1) size(m,2)/2]);
          m_twice=imresize(m,[size(m,1) size(m,2)*2]);
          data_augment_1{i,1}=m_half;
          data_augment_2{i,1}=m_twice;
    end  
      train_audio=cat(1,train_audio,data_augment_1,data_augment_2);% add augmented to the training data
      train_label=repmat(train_label,[3 1]);
end
  
% for obtaining fixed weight from CNN
% c=load('20150923_task1_pca.mat');
% f=squeeze(c.params{2}.weight);
% clear train_audio
% for i=1:size(f,3)
%     train_audio{i}=f(:,:,i);
% end 
% train_audio=train_audio';

%% STEP 3. Extract feature from pretrained CNNs --- [DO NOT MODIFY]
% In this step we integrated features from each time step into one
% fixed-length feature
[train_audio, valid_audio, test_audio] = global_pooling(train_audio, valid_audio, test_audio, opt.pooling_type);


%% STEP 4-0. Training MLPs --- [DO NOT MODIFY]
% Initialize some parameters of 2-layers MLP



mlp(2) = struct();
relu(2) = struct();

mlp(1).input = [];
mlp(1).error = [];
mlp(1).weight = opt.init_std * randn(size(train_audio, 2), opt.hidden_size);
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
    
    input = train_audio(cur_batch, :);
 
    for j = 1:2
        
        [mlp(j), input] = mlp_feedforward(mlp(j), input, 'fc');
        if j == 1
            [relu(j), input] = mlp_feedforward(relu(j), input, 'relu');
        end
    end
    
    %% STEP 5. Softmax Regression --- [COMPLETE CODE FOR THIS STEP]
    % Implement softmax_regression.m
    [nll, ~, err] = softmax_regression(input, train_label(cur_batch, :));
%     while 1
%         disp('stop me');
%     end
%     
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
        %disp('hi');
        mlp(j) = weight_update(mlp(j), opt.learning_rate, opt.batch_size);
    end
    
    if rem(i, opt.print_train_freq) == 0
        fprintf('(%d / %d)  iteration, Error = %f\n', i, opt.total_iteration, nll)
    end
    

    if rem(i, opt.check_valid_freq) == 0
        [valid_nll, valid_acc] = evaluate(valid_audio, valid_label, {mlp, relu});
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
[test_nll, test_acc] = evaluate(test_audio, test_label, model);

fprintf('Test, Error = %.2f, Accuracy = %.2f%%\n', ...
            test_nll, 100 * test_acc)
%}