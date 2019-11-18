clc;
close all;

addpath('/home/asthemus/speaker-recognition/lib');
global_path = '/home/asthemus/speaker-recognition/lib/';

%% test phase

% % data writing
d = 'eval';

% directory reading
test_dir = strcat(global_path,'test/');
test_files = dir(test_dir);
test_files = test_files(3:end);
for k=1:length(test_files)
        %test_file_names=setdiff({test_files(k).name},{'.','..'});
        
        %file_loc = strcat(test_dir,test_file_names);
        test_audio_file=[];
            test_audio_data{k} = audioread(file_loc{1});
            [file_name{k} fff] = regexp(test_files(k).name, '\.', 'split');   
end   

crop = 8000;
frame_length = 2400;
threshold = 0.01;
x = 1;

% % silent frames removing
for i= 1 : length(test_audio_data)
  test_audio_data{i} = test_audio_data{i}(crop : (length(test_audio_data{i}) - crop + frame_length));
    
  N = length(test_audio_data{i});
  frame_count = floor(N / frame_length);
    
  for k= 1 : frame_count
    frame = test_audio_data{i}((k-1)*frame_length + 1 : frame_length * k);
    max_range = max(frame);
      
    if (max_range > threshold)
      test_signals{i}((x-1)*frame_length + 1 : frame_length * x) = frame;
      x = x + 1;
      clear frame;
    end
  end
  x = 1;
end

% % MFCC feature extraction
for i= 1 : length(test_signals)
  test_data{i} = mfcc(test_signals{i}, 400, 240, Fs, 23, l, nceps, hamming(64), R, alpha);
end

% get posterior

for i=1:length(test_data)
    disp(strcat('getting posterior for data ',int2str(i)));
    p=[];
    for k=1:10
        p = [p, pdf(GMModel{k}, transpose(test_data{i}))];% P(x|model_i)
    end
    [~, cIdx] = max(p,[],2); 
    figure
    step = Fs * 10 / 1000;
    subplot(2, 1, 1)
    plot(1:step:step*size(cIdx, 1), p);
    legend('class 1', 'class 2','class 3', 'class 4','class 5', 'class 6','class 7', 'class 8','class 9', 'class 10');
    subplot(2, 1, 2)
    plot(test_signals{i} + 1.5);
end

% argmax_i P(x|model_i)

% % creating a file
% fileID = fopen('audio_GMM.txt', 'at');
% formatSpec = '%s %d %f\n';
%   
% for i= 1 : length(test_data)
%   for k= 1 : 10
%     % class evaluation
%     ll(k) = sum(logpdf_gmm(test_data{i}, gauss_weight{k}, gauss_meanv{k}, gauss_covm{k}));
%   end;
%   % final class decision  
%   [maximum_ll, index] = max(ll);
%   
%   fprintf(fileID, formatSpec, file_name{i}{1}, index, maximum_ll);                
% end;                                  
% 
% fclose(fileID);