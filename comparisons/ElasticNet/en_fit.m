function [Z_result,g_result] = en_fit(data,varargin)

    %% Parse data and optional inputs
    parser = inputParser();
    addOptional(parser,'tau',[0])
    addOptional(parser,'A_re',[0])
    addOptional(parser,'A_im',[0])
    addOptional(parser,'induc',0)
    addOptional(parser,'freq',[0])
    parse(parser,varargin{:})

    f = data.Freq;
    
    % if tau not specified, use 1/(2*pi*f)
    if length(parser.Results.tau)==1
        t = 1./(f*2*pi);
    else
        t = parser.Results.tau;
    end 
    % need ln(tau) increment for scaling DRT
    dlnt = log(t(2)) - log(t(1));
    
    % check if precomputed A_re and A_im given
    s_re = size(parser.Results.A_re);
    s_im = size(parser.Results.A_im);
    if (s_re(1)==1 && s_re(2)==1) || (s_im(1)==1 && s_im(2)==1)
        disp('Calculating A matrices...')
        [A_real,A_imag] = cal_Basis(f,t);
        disp('Finished calculating A matrices')
    else
        A_real = parser.Results.A_re;
        A_imag = parser.Results.A_im;
    end
    
    % impedance data
    Z_real = data.Zreal;
    Z_imag = data.Zimag;

    %% Deconvolution
    lambda = logspace(-10,1,100); %set-up grid of shrinkage tuning parameter
    model = sms_DRT(Z_real,Z_imag,A_real,A_imag,lambda,parser.Results.induc,parser.Results.freq);

    %% Estimation results
    R_infy_est = model.R_infy;
    R_p_est = model.R_p;
    DRT_est = model.beta;
    Z_real_est = model.Z_real;
    Z_imag_est = model.Z_imag;

    %% Output tables
    Z_result = array2table([f Z_real_est Z_imag_est],...
        'VariableNames',{'freq' 'Zreal' 'Zimag'});

    tc = (t(2:end) + t(1:end-1))./2; % using center of inteval [t_m, t_{m+1})
    g_result = array2table([tc DRT_est.*R_p_est./dlnt], 'VariableNames',{'tau' 'gamma'});

end

