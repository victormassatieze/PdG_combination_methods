function [final_TFR, masked, fs, Win_matrix, phase_tensor, spec_tensor, hop_size, mirror_length] = runFHLS(path, ...
    resolutions)

    if (nargin == 1)
        resolutions = [4096, 2048, 1024];
    end
    
    [signal,fs] = audioread(path);
    if (fs ~= 48e3)
        [P,Q] = rat(48e3/fs);
        signal = resample(signal,P,Q);
        fs = 48e3;
        audiowrite(path, signal, 48e3);
    end
    if (size(signal,2) ~= 1)
        signal = signal(:,1);
        audiowrite(path, signal, 48e3);
    end
    
    n_fft = max(resolutions);
    if mod(n_fft,2) == 0
        n_fft = n_fft+1;
    end
    hop_size = round(min(resolutions)/4);

    zeros1 = zeros(round((n_fft-resolutions(1)-1)/2),1) + min(hamming(resolutions(1)+1));
    win1 = [zeros1; hann(resolutions(1)+1); zeros1];
    %win1 = [hann(resolutions(1)+1); zeros1; zeros1];
    zeros2 = zeros(round((n_fft-resolutions(2)-1)/2),1) + min(hamming(resolutions(2)+1));
    win2 = [zeros2; hann(resolutions(2)+1); zeros2];
    %win2 = [hann(resolutions(2)+1); zeros2; zeros2];
    zeros3 = zeros(round((n_fft-resolutions(3)-1)/2),1) + min(hamming(resolutions(3)+1));
    win3 = [zeros3; hann(resolutions(3)+1); zeros3];
    %win3 = [hann(resolutions(3)+1); zeros3; zeros3];
    
    mirror_length = (n_fft - 1)/2;
    pad = signal(mirror_length:-1:1);
    mirror_pad = [pad; signal];
    clear signal
    signal = mirror_pad;

    Win_matrix = [win1, win2, win3];

    X1 = stft([signal; zeros(max(resolutions),1)],fs,'Window',win1,...
        'OverlapLength',n_fft-hop_size,'FFTLength',n_fft,...
        'FrequencyRange','onesided');
    X2 = stft([signal; zeros(max(resolutions),1)],fs,'Window',win2,...
        'OverlapLength',n_fft-hop_size,'FFTLength',n_fft,...
        'FrequencyRange','onesided');
    X3 = stft([signal; zeros(max(resolutions),1)],fs,'Window',win3,...
        'OverlapLength',n_fft-hop_size,'FFTLength',n_fft,...
        'FrequencyRange','onesided');

    spec_tensor(:,:,1) = abs(X1);
    phase_tensor(:,:,1) = angle(X1);
    spec_tensor(:,:,2) = abs(X2);
    phase_tensor(:,:,2) = angle(X2);
    spec_tensor(:,:,3) = abs(X3);
    phase_tensor(:,:,3) = angle(X3);

    size_w_S = [17, 13];
    eta = 8;

    [final_TFR, masked] = spectrogram_comb_FastHoyerLocalSparsity(spec_tensor, size_w_S, eta);

end