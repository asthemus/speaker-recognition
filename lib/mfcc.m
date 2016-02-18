function CPS = mfcc(S, WINDOW, NOVERLAP, NFFT, Fs, NBANKS, NCEPS)
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
%
%   See also SPECTROGRAM


% Add low level noise (40dB SNR) to avoid log of zeros 
SNRdB = 40;
noise = rand(size(S));
norm(S) / norm(noise) / 10^(SNRdB/20);
S = S + noise * norm(S) / norm(noise) / 10^(SNRdB/20);

% Initialize matrices representing Mel filterbank and DCT
mfb = mel_filter_bank(NFFT, NBANKS, Fs, 32); % first filer starts at 32Hz
dct_mx = dct(eye(NBANKS));

CPS = dct_mx(1:NCEPS,:) * log(mfb' * abs(spectrogram(S, WINDOW, NOVERLAP, NFFT, Fs)));