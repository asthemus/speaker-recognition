clc;
close all;

addpath('/home/asthemus/speaker-recognition/lib');
%x1 = audioread('/home/asthemus/speaker-recognition/lib/train/id10001/0001.wav');
%x1 = transpose(x1);
%global_data{1}{1}  = x1;
%x2 = audioread('/train/id10001/00002.wav');
%x2 = transpose(x2);
%global_data{1}{2} = x2;
%x3 = audioread('/train/id10001/00003.wav');
%x3 = transpose(x3);
%global_data{1}{3} = x3;
%f=[];
%f = [f, x1];
%f = [f, x2];
%f = [f, x3];
%global_f{1} = f;


%x4 = audioread('/train/id10001/0004.wav');
%x4 = transpose(x4);
%x5 = audioread('/train/id10001/00004.wav');
%x5 = transpose(x5);
%global_data{2}{1} = x4;
%global_data{2}{2} = x5;
%g=[];
%g = [g, x4];
%g = [g, x5];
%global_f{2} = g;

dir_start  = 'id1';

for i=1:10   
     id_padded = sprintf('%04d',i);
     dir_name = strcat(dir_start, id_padded);
     disp(strcat('checking directory ', dir_name));
     D = strcat('/home/asthemus/speaker-recognition/lib/train/',dir_name);
     sub_files = dir(D);
     
     for k=1:length(sub_files)
        file_names=setdiff({sub_files(k).name},{'.','..'});
        file_loc = strcat(D,'/',file_names);
        disp(file_loc);
        user_audio_file=[];
        if(~isempty(file_loc))
            user_audio_file = audioread(file_loc{1});
            global_data{i}{k} = user_audio_file;
        end    
     end
     
     global_data{i} = global_data{i}(3:end);
end

    
% data pre-processing 
crop = 8000;
frame_length = 2400;
threshold = 0.01;
l=22;
Fs = 44100;
R = [300 3700];
alpha=0.97;
nceps =13;
x = 1;

N_GMM = 16;     % num of GMM classes
N_sample = 10;  % number of samples (total)
N_train = 9;   % number of training samples to use, should be less than 20
        

for i= 1 : length(global_data)
  for j= 1 : length(global_data{i})
    
    % acustic artefact removing
    global_data{i}{j} = global_data{i}{j}(crop : (length(global_data{i}{j}) - crop + frame_length));
    
    % silence removing
    N = length(global_data{i}{j});
    frame_count = floor(N / frame_length);
    
    for k= 1 : frame_count
      frame = global_data{i}{j}((k-1)*frame_length + 1 : frame_length * k);
      max_range = max(frame);
      
      % thresholding
      if (max_range > threshold)
        signals{i}{j}((x-1)*frame_length + 1 : frame_length * x) = frame;
        x = x + 1;
        clear frame;
      end
    end
    %plot(signals{i}{j});
    %pause;      
    x = 1;
  end
end
    
%MFCC features extraction
for i= 1 : 10
  for j= 1 : length(signals{i})
    [features{j},~,~] = mfcc(signals{i}{j}, 400, 240, Fs, 23, l, nceps, hamming(64), R, alpha);
  end
  train_data{i} = cell2mat(features);
end

% After getting the feature, we train a GMM model, N_GMM

% GMM Training
tic
options = statset('MaxIter', 500);
for i=1:10
    GMModel{i} = fitgmdist(MFCCs1, N_GMM, 'Options', options);
toc


% % number of training iterations
% train_count = 8;
% 
% for i= 1 : 10
%   % number of Gaussian Mixtures
%   gauss_count{i} = 5;
%   
%   % expected value calculation 
%   mtd = train_data{i};
%   x = floor(length(mtd)/4); 
%   y = floor(length(mtd)/2);
%   z = floor((3 * length(mtd))/4);
%   
%   for n= 1 : nceps
%     mean_vector(n,1) = mean(mean(mtd(n,:)));
%     mean_vector(n,2) = mean(mean(mtd(n,(1:x))));
%     mean_vector(n,3) = mean(mean(mtd(n,(z:end))));
%     mean_vector(n,4) = mean(mean(mtd(n,(x:y))));
%     mean_vector(n,5) = mean(mean(mtd(n,(y:z))));
%   end
%   
%   gauss_meanv{i} = mean_vector;  
% 
%   % covariance matrix
%   gauss_covm{i} = repmat(var(train_data{i}', 1)', 1, gauss_count{i});
% 
%   % weighting vector
%   gauss_weight{i} = ones(1, gauss_count{i}) / gauss_count{i};      
%   
%   % training process
%   for j=1 : train_count
%     [gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}, ttl{i}] = train_gmm(train_data{i}, gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}); 
%   end
%   
%   disp(['Train data: ' num2str(i)])
%   disp(['  Total log-likelihood: ' num2str(ttl{i})])
% end

% % data writing
% d = 'eval';
% clear f;
% 
% % directory reading
% test_dir_name = strcat(d, '/*.wav');
% test_files = dir([test_dir_name]);
% 
% for i= 1 : length(test_files)
%   f = strcat(d, '/', test_files(i).name);
%   disp(f);
%   test_wav_data{i} = wavread([f]);
%   [file_name{i} fff]= regexp(test_files(i).name, '\.', 'split');   
% end;    
% 
% crop = 8000;
% frame_length = 2400;
% threshold = 0.01;
% x = 1;
% 
% % silent frames removing
% for i= 1 : length(test_wav_data)
%   test_wav_data{i} = test_wav_data{i}(crop : (length(test_wav_data{i}) - crop + frame_length));
%     
%   N = length(test_wav_data{i});
%   frame_count = floor(N / frame_length);
%     
%   for k= 1 : frame_count
%     frame = test_wav_data{i}((k-1)*frame_length + 1 : frame_length * k);
%     max_range = max(frame);
%       
%     if (max_range > threshold)
%       test_signals{i}((x-1)*frame_length + 1 : frame_length * x) = frame;
%       x = x + 1;
%       clear frame;
%     end;
%   end;
%   x = 1;
% end;
% 
% % MFCC feature extraction
% for i= 1 : length(test_signals)
%   test_data{i} = mfcc(test_signals{i}, 400, 240, 512, 16000, 23, 13);
% end;
% 
% % creating a file
% fileID = fopen('audio_GMM.txt', 'at');
% formatSpec = '%s %d %f\n';
%   
% for i= 1 : length(test_data)
%   for k= 1 : 31
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