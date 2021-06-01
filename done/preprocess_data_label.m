function preprocess_data_label(data_dir, output_data_dir)


data_audio = load_data(data_dir, 'audio');
data_video = load_data(data_dir, 'video');
train_num = size(data_audio.train_data, 1);     % number of the training data
valid_num = size(data_audio.valid_data, 1);     % number of the validation data
test_num = size(data_audio.test_data, 1);     % number of the test data

%% STEP 2. Audio pre-processing

% pre-processing parameters [DO NOT MODIFY]
opt.fs = 48000;             % frequency sampling rates
opt.frameLenMS = 25;        % Frame length in ms
opt.frameShiftMS = 10;      % Frame shift in ms
opt.preemph = 0.97;         % Pre-emphasize parameter
opt.minfreq = 0;           % Minimum frequency
opt.maxfreq = 8000;         % Maximum frequency
opt.nfilts = 26;            % Number of Mel filters
opt.draw_figure = false;

% extracting log-mel spectrogram
train_logmel = cell(train_num,1);
for idx = 1:train_num
    train_logmel{idx,1} = filterbank(data_audio.train_data{idx,1}, opt);                    %% Task 1. fill in FILL_filterbank.m
end
valid_logmel = cell(valid_num,1);
for idx = 1:valid_num
    valid_logmel{idx,1} = filterbank(data_audio.valid_data{idx,1}, opt);                      %% Task 1. fill in FILL_filterbank.m
end
test_logmel = cell(test_num,1);
for idx = 1:test_num
    test_logmel{idx,1} = filterbank(data_audio.test_data{idx,1}, opt);                    %% Task 1. fill in FILL_filterbank.m
end
audio_logmel = {train_logmel, valid_logmel, test_logmel};
save(fullfile(output_data_dir, 'audio_processed.mat'), 'audio_logmel');

%% STEP 3. Image pre-processing

patch_width=48;   % width of the patches [DO NOT MODIFY]
patch_height=96;  % height of the patches [DO NOT MODIFY]

train_video = normalize_image(data_video.train_data, patch_width, patch_height);   
valid_video = normalize_image(data_video.valid_data, patch_width, patch_height);   
test_video = normalize_image(data_video.test_data, patch_width, patch_height);   
video_processed = {train_video, valid_video, test_video};
%save(fullfile(output_data_dir, 'video_processed.mat'), 'video_processed');


%% Save data labels
label = {data_audio.train_label, data_audio.valid_label, data_audio.test_label};
save(fullfile(output_data_dir, 'data_label.mat'), 'label');

end
