function [ CPS, FBE, frames] = mfcc(S, WINDOW, NOVERLAP, Fs, NBANKS, LIFT, NCEPS, type_window, R, alpha)
%MFCC Mel Frequency Cepstral Coefficients
%   CPS = MFCC(s, FFTL, Fs, WINDOW, NOVERLAP, NBANKS, NCEPS) returns 
%   NCEPS-by-M matrix of MFCC coeficients extracted form signal S, where
%   M is the number of extracted frames, which can be computed as
%   floor((length(S)-NOVERLAP)/(WINDOW-NOVERLAP)). Remaining parameters
%   have the following meaning:
%
%   NFFT          - number of frequency points used to calculate the discrete
%                   Fourier transforms
%   Fs            - sampling frequency [Hz]
%   WINDOW        - window lentgth for frame (in samples)
%   NOVERLAP      - overlapping between frames (in samples)
%   NBANKS        - numer of mel filter bank bands
%   NCEPS         - number of cepstral coefficients - the output dimensionality
%   LIFT          - liftering parameter
%   type_window   - is a analysis window function handle
%   ALPHA         - preemphasis coefficient
%   See also SPECTROGRAM


% Add low level noise (40dB SNR) to avoid log of zeros 
SNRdB = 40;
noise = rand(size(S));
norm(S) / norm(noise) / 10^(SNRdB/20);
S = S + noise * norm(S) / norm(noise) / 10^(SNRdB/20);

% Explode samples to the range of 16 bit shorts
if( max(abs(S))<=1 ), S = S * 2^15; end

nfft = 2^nextpow2(WINDOW);     % length of FFT analysis 
K = nfft/2+1;                % length of the unique part of the FFT

%% HANDY INLINE FUNCTION HANDLES

hz2mel = @( hz )( 1127*log(1+hz/700) );     % Hertz to mel warping function
mel2hz = @( mel )( 700*exp(mel/1127)-700 ); % mel to Hertz warping function

% Type III DCT matrix routine (DCT - Discrete Cosine Transform)
dctm = @( NCEPS, NBANKS)(sqrt(2.0/NBANKS)*cos(repmat([0:NCEPS-1].',1,NBANKS).*repmat(pi*([1:NBANKS]-0.5)/NBANKS,NCEPS,1)));

% Cepstral lifter routine 
ceplifter = @( NCEPS, LIFT )( 1+0.5*LIFT*sin(pi*[0:NCEPS-1]/LIFT) );

%% FEATURE EXTRACTION 

S = filter( [1 -alpha], 1, S ); % fvtool( [1 -alpha], 1 ); First order FIR Filter

% Framing and windowing (frames as columns)
frames = vec2frames(S, WINDOW, NOVERLAP, 'cols', type_window, false );

% Magnitude spectrum computation (as column vectors)
MAG = abs( fft(frames,nfft,1) ); 

% Triangular filterbank with uniformly spaced filters on mel scale
H = trifbank(NBANKS, K, R, Fs, hz2mel, mel2hz ); % size of H is M x K 

% Filterbank application to unique part of the magnitude spectrum
FBE = H * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor

% DCT matrix computation
DCT = dctm( NCEPS, NBANKS );

% Conversion of logFBEs to cepstral coefficients through DCT
CPS =  DCT * log( FBE );

% Cepstral lifter computation
lifter = ceplifter( NCEPS, LIFT );
     
% Cepstral liftering gives liftered cepstral coefficients
CPS = diag( lifter ) * CPS; % ~ HTK's MFCCs

% Initialize matrices representing Mel filterbank and DCT
%mfb = mel_filter_bank(NFFT, NBANKS, Fs, 32); % first filer starts at 32Hz
%dct_mx = dct(eye(NBANKS));

%CPS = dct_mx(1:NCEPS,:) * log(mfb' * abs(spectrogram(S, WINDOW, NOVERLAP, NFFT, Fs)));