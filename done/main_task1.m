
% EE476_Audio_visual_perceptron_model
% Homework 3
%
% main_task1.m
% For audio data, build the CNN architecture and train weights by PCA and sparse AE
% CNN architecture trained by PCA is provided.
% TASK 1A : write a function which trains sparse AE with the training data
% TASK 1B : propagate to convolution layer in which weights are trained by sparse AE
% TASK 1C : propagate to pooling layer

%% SETP 0.
 close all; rng(0);%clear all;
student_id = 20150923;
your_name = 'your_name';
fprintf('[Homework 3-1] name : %s, student id : %d \n', your_name, student_id ) % [DO NOT MODIFY]

% load pre-processed data
train_audio_dir = '../data/train/audio';
train_video_dir = '../data/train/video';
train_prep_audio_dir='../data/train/audio_logmel.mat';
train_prep_video_dir='../data/train/video_prep.mat';

if ~exist(train_prep_audio_dir, 'file')
    [train_audio, train_video] = preprocess_data(train_audio_dir, train_video_dir, train_prep_audio_dir, train_prep_video_dir);
else
    train_audio=load(train_prep_audio_dir, 'logmel');
    train_audio=train_audio.logmel;
end

num_data = size(train_audio,1);     % number of the training data
num_patch = 30000;                  % number of patch

fprintf('Load data .. done \n');

%% CNN architecture
% feature extraction
feat_ext = 'pca'; % or 'sae'
fprintf(sprintf('Feature extraction method : %s \n', feat_ext));  % [DO NOT MODIFY]

% configurations [DO NOT MODIFY]
config = cell(2,1);
config{1,1}.in_height=26;
config{1,1}.in_feat_maps=1;
config{1,1}.filter_height=26;
config{1,1}.filter_width=5;
config{1,1}.out_feat_maps=48;
config{1,1}.pool=2;

config{2,1}.in_height=config{1,1}.in_height-config{1,1}.filter_height+1;
config{2,1}.in_feat_maps=config{1,1}.out_feat_maps;
config{2,1}.filter_height=1;
config{2,1}.filter_width=5;
config{2,1}.out_feat_maps=96;
config{2,1}.pool=2;

params = cell(2,1);

for layer = 1:2
    % cropping patches from input data
    data_audio = zeros(config{layer,1}.in_height*config{layer,1}.filter_width*config{layer,1}.in_feat_maps, num_patch);
    for idx=1:num_patch
        data_idx = ceil(rand(1,1)*num_data);
        data = train_audio{data_idx,layer};
        nframe = size(data,2);
        frame = ceil(rand(1,1)*(nframe-config{layer,1}.filter_width+1));
        data_audio(:,idx) = reshape(data(:, frame:frame+config{layer,1}.filter_width-1, :), ...
            config{layer,1}.in_height * config{layer,1}.filter_width * config{layer,1}.in_feat_maps, 1);
    end
    
    if(strcmp(feat_ext,'pca'))
        
        % feature extraction
        [pc, m, v] = pca2(data_audio);
        weight = pc(:, 1:config{layer,1}.out_feat_maps);
        
        params{layer,1}.weight = reshape(weight, config{layer,1}.in_height, config{layer,1}.filter_width ,config{layer,1}.in_feat_maps, config{layer,1}.out_feat_maps);
        params{layer,1}.bias = reshape(m, config{layer,1}.filter_height, config{layer,1}.filter_width, config{layer,1}.in_feat_maps);
        
        % propagate data to the convolution and pooling layers
        for data_idx = 1:num_data
            
            data = train_audio{data_idx,layer};
            
            % convolution
            activation_c = zeros(config{layer,1}.in_height - config{layer,1}.filter_height + 1, ...
                size(data,2) - config{layer,1}.filter_width + 1 , ...
                config{layer,1}.out_feat_maps); % convolution activation
            
            for k=1:config{layer,1}.out_feat_maps
                for j=1:size(data,2) - config{layer,1}.filter_width + 1
                    for i=1:config{layer,1}.in_height - config{layer,1}.filter_height + 1
                        patch = reshape(data(:, j:j+config{layer,1}.filter_width-1, :), ...
                            config{layer,1}.in_height * config{layer,1}.filter_width * config{layer,1}.in_feat_maps, 1);
                        activation_c(i,j,k) = weight(:,k)' * (patch - m);
                    end
                end
            end
            
            % pooling and non-lienar function
            activation_p = zeros(size(activation_c,1), ...
                floor(size(activation_c,2)/2), ...
                size(activation_c,3)); % pooling activation
            
            for k=1:size(activation_p,3)
                for j=1:size(activation_p,2)
                    for i=1:size(activation_p,1)
                        activation_p(i,j,k) = max(activation_c(i, (j-1)*config{layer,1}.pool+1:j*config{layer,1}.pool, k));
                        activation_p(i,j,k) = max(0, activation_p(i,j,k)); %relu non-linear function
                    end
                end
            end
            train_audio{data_idx, layer+1} = activation_p;
            
        end
        
    elseif (strcmp(feat_ext,'sae'))
        % feature extraction
        %-- TASK 1A : write a function 'train_sae.m' which trains sparse AE with 'data_audio'
        %-- you can utilize the part of codes or modules in all previous homework
        %-- you can also define 'sae_config' for nIn, nOut,
        %-- nHidden, and all other training hyperparameters as you wish.
        sae_config.layer = layer;
        sae_config.nHidden = config{layer,1}.out_feat_maps; % the number of hidden neurons.
        sae_config.lRate = 0.01; % learning rate
        sae_config.use_sparsity = true; % option to control use of sparsity term in training autoencoder
        sae_config.AEepoch = 50000; %50000 the number of epoch to train Autoencdoer. Modify this only if you think training 50000epoch is not enough.
        sae_config.sparsity_target = 0.1; % Sparsity target: target activation for average hidden neuron values
        sae_config.sparsity_coeff = 10; % Sparsity coefficients: how much do you want to weigh sparsity learning compared to reconstruction
        
        [weight, bias] = train_sae(data_audio, sae_config); 
        
        params{layer,1}.weight = weight; % resize it as proper size
        params{layer,1}.bias = bias; % resize it as proper size
        
        % propagate data to the convolution and pooling layers
        
        for data_idx = 1:num_data
            
            data = train_audio{data_idx,layer};
            
            % convolution
            activation_c = zeros(config{layer,1}.in_height - config{layer,1}.filter_height + 1, ...
                size(data,2) - config{layer,1}.filter_width + 1 , ...
                config{layer,1}.out_feat_maps); % convolution activation
            
            %%-- TASK 1B : fill here --%%
            %%- you can use 'conv', 'conv2', or 'convn' functions for
            %%- efficient calcuation 
            for j=1:size(data,2) - config{layer,1}.filter_width + 1
                for i=1:config{layer,1}.in_height - config{layer,1}.filter_height + 1
                    patch = reshape(data(:, j:j+config{layer,1}.filter_width-1, :), ...
                        config{layer,1}.in_height * config{layer,1}.filter_width * config{layer,1}.in_feat_maps, 1);
                    activation_c(i,j,:) = weight * patch + bias;
                end
            end
            % pooling and non-lienar function
            activation_p = zeros(size(activation_c,1), ...
                floor(size(activation_c,2)/2), ...
                size(activation_c,3)); % pooling activation
            
            %%-- TASK 1C : fill here --%%
            for k=1:size(activation_p,3)
                for j=1:size(activation_p,2)
                    for i=1:size(activation_p,1)
                        activation_p(i,j,k) = max(activation_c(i, (j-1)*config{layer,1}.pool+1:j*config{layer,1}.pool, k));
                    end
                end
            end
            
            train_audio{data_idx, layer+1} = activation_p;
            
        end
    end
end

% save trained weights [DO NOT MODIFY]
file_name = sprintf('%d_task1_%s.mat', student_id, feat_ext);
fprintf(sprintf('Save the trained weights to ... : %s \n',file_name));
save(file_name, 'params');














