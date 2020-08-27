%% Set parameters for A matrices
% for basic simulated spectra
freq = logspace(6,-2,81)';
tau = logspace(-7,2,200)';
A_file = 'A_f6,-2,81_t-7,2,200.mat';

% for truncated simulated spectrum
freq2 = logspace(4.5,-0.5,51)';
tau2 = logspace(-7,2,200)';
A_file2 = 'A_f4.5,-0.5,51_t-7,2,200.mat';

%% Load and fit simulated impedance files
% basic spectra
[A_re,A_im] = get_A(A_file,freq,tau);
data_path = '../../data';
sim_files = dir(strcat(data_path,'/simulated/Z*.csv'));
% remove trunc files
sim_files = sim_files(~contains({sim_files.name},'trunc'));
fit_files(sim_files,A_re,A_im,tau,0) % no inductance

% truncated spectrum
[A_re,A_im] = get_A(A_file2,freq2,tau2);
trunc_files = dir(strcat(data_path,'/simulated/Z_trunc*.csv'));
fit_files(trunc_files,A_re,A_im,tau2,1) % fit inductance

function [A_re,A_im] = get_A(A_file,f,tau)
    if isfile(A_file)
        A_load = load(A_file);
        A_re = A_load.A_re;
        A_im = A_load.A_im;
    else
        % if precomputed files not available, calculate matrices
        disp('Calculating A matrices...')
        [A_re,A_im] = cal_Basis(f,tau);
        save(A_file,'A_re','A_im')
        disp('Finished calculating matrices')
    end
end

function fit_files(files,A_re,A_im,tau,induc)
    for n = 1:length(files)
        file = files(n);
        filepath = strcat(file.folder,'/',file.name);
        data = readtable(filepath);
        % fit data
        [Z_res,g_res] = en_fit(data,tau,A_re,A_im,induc,data.Freq);
        % save results
        suffix = file.name(2:end);
        writetable(Z_res,strcat('results/Zout',suffix))
        writetable(g_res,strcat('results/Gout',suffix))
    end
end