function [beta1_i, Ci] = MichaelSpatial2(Y,Runs,nuis)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% INPUTS:
%
% Y:    Data matrix (#time points) by (#regions)
% Runs: Stick functions for each subject (#time points) by (#conditions) 
%


% Indices:
%
% Subjects: i = 1,.... n
% Time: t=1,.... T
% Stimulus: j=1,.... J
% Voxel: V_b = 1,.... Vb
% Regions: b=1,.... B


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values

[T, regions] = size(Y);         % # time points and subjects
[~, J] = size(Runs);            % # conditions


TR = 1;                         % TR of experiment
Tlen = 30/TR;                   % Length of HRF

K = 15;                         % Number of b-spline basis
norder = 6;                     % Order of b-spline basis

q = 2;                          % Number of nuisance parameters
% p = 1;                        % AR order


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate basis sets for potential HRF shapes.

% Create bspline basis set
basis = create_bspline_basis([0,Tlen], K+8, norder);    
B = eval_basis((1:Tlen),basis);
B = B(:,4:end-5);

Wi = zeros(T, J*K);
for j=1:J  
    Wji = tor_make_deconv_mtx3(Runs(:,j),Tlen,1);    
    Wi(:,(j-1)*K+1:j*K) = Wji(:,1:Tlen)*B;    
end

X1 = [ones(T,1) Wi];                    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nuisance Model

X2 =ones(T,q);

for l = 1:q    
    tmp = (1:T).^l;
    tmp = tmp/max(tmp);
    X2(:,l) = tmp;
end
X2 = [X2 nuis];  

beta1_i = zeros(regions,J*K+1); 
rho_i = zeros(regions,1);
sig_i = zeros(regions,1);
E_i = zeros(regions,T);


tmp = 0;

for r = 1:regions
    
    [beta1_ir, ~, rho_ir, sig_ir, resid] = ROIfit(Y(:,r),X1,X2);
 
    beta1_i(r,:) = beta1_ir';
    rho_i(r) = -rho_ir(2);
    sig_i(r) = sig_ir;
    E_i(r,:) = resid';
    
end

Ci = zeros(regions*(J*K+1),regions*(J*K+1));


XXX = X1*inv(X1'*X1);
phi_i = (1-rho_i*rho_i').*(E_i*E_i')./T;

ARmats = [];
tp=1:T;
for j=1:regions,
    tmp = zeros(2*T,T);
    for t=1:T,
        tmp(t+1:t+T,t) = rho_i(j).^(tp);
    end
    
    ARmats{j} = tmp(1:T,:);
    
end


for j=1:regions
    for k=1:regions
        
        Sjk =eye(T,T);
        
        Sjk = Sjk + ARmats{j} + ARmats{k}';    
        
        Sjk = (phi_i(j,k)/(1-rho_i(j)*rho_i(k)))*Sjk; 
        
        Ci((j-1)*(J*K+1)+1:(j*(J*K+1)),(k-1)*(J*K+1)+1:(k*(J*K+1))) = XXX'*Sjk*XXX;
        
        disp(k)
    end;
    

end

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   SUBFUNCTIONS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [beta1, beta2, rho, sig2, resid] = ROIfit(Y,X1, X2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% INPUTS:
%
% Y:    Data matrix (#time points) by (#voxels)
% Runs: Stick functions for each subject (#time points) by (#conditions) 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit Model

% Compute region-specific time course

p1 = size(X1,2);
p2 = size(X2,2);

Ybar = mean(Y,2); 
X = [X1 X2];
beta = pinv(X)*Ybar;
resid = Ybar - X*beta;
[rho,sig2]=aryule(resid,1);
rho = -rho;

beta1 = beta(1:p1);
beta2 = beta(p1+1:end);


end




