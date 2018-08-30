%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load data from BMRK3

load('n33_buckner17_k286.mat');
load('wani_33_variables.mat')


% Add paths

addpath('Matlabfunctions');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note:
%
% Also need SPM8 (http://www.fil.ion.ucl.ac.uk/spm/software/)and
% CanlabCore tools (https://github.com/canlab/CanlabCore) added to the
% path.



%%

% Set-up

T = 1845;                       % Number of time points
regions = 286;                  % Number of regions

J=2;                            % Number of conditions

TR = 1;                         % TR of experiment
Tlen = 30/TR;                   % Length of HRF

K = 15;                         % Number of b-spline basis
norder = 6;                     % Order of b-spline basis

% Create bspline basis set
basis = create_bspline_basis([0,Tlen], K+8, norder);    
B = eval_basis((1:Tlen),basis);
B = full(B(:,4:end-5));

% num = size(B,2);

%%

% Create HRFs

TR = 2;
len = round(30/TR);
xBF.dt = TR;
xBF.length= len;
xBF.name = 'hrf (with time and dispersion derivatives)';
xBF = spm_get_bf(xBF);

v1 = xBF.bf(1:len,1);
v2 = xBF.bf(1:len,2);
v3 = xBF.bf(1:len,3);

h = v1;
dh =  v2 - (v2'*v1/norm(v1)^2).*v1;
dh2 =  v3 - (v3'*v1/norm(v1)^2).*v1 - (v3'*dh/norm(dh)^2).*dh;

h = h./max(h);
dh = dh./max(dh);
dh2 = dh2./max(dh2);

%%

% Run spatial model for each subject/ save in Results_sub**.mat

TR = 1;

for sub = 1:33,

    Y = n33_buckner17_k286{sub};
    Runs = zeros(T,2);
    Runs(round(wani_33_model4.warmonset{sub}/2),1) = 1;
    Runs(round(wani_33_model4.painonset{sub}/2),2) = 1;
    
    covrun = zeros(T,1);
      
    for i=1:97,
        tmp = wani_33.reportonset(:,sub);
        covrun(round(tmp(i)/2)) = 1;
    end
    
    
    nuis = [conv(covrun,h) conv(covrun,dh) conv(covrun,dh2)];
    nuis = nuis(1:T,:);
 
    [beta1_i, Ci] = MichaelSpatial2(Y,Runs,nuis);
     
    name = strcat('Results_sub',num2str(sub));
    
    save(name,'beta1_i','Ci')

end


%%

num = regions*(K*J+1);

ind = [1 4:6 9:10 13 15:21 23:26 28:33];
nsub = length(ind);


C = [];
Bm = [];        % beta in matrix format
Bv = [];        % beta in vector format
invCi = [];
B_new = 0;


for i=ind
    
    name = strcat('Results_sub',num2str(i));
    load(name)
    C{i} = Ci;
    Bm{i} = beta1_i;
    Bv{i} = reshape(Bm{i}',num,1);
    invCi{i} = inv(Ci);

    B_new = B_new + Bv{i}./nsub;
    
    disp(i)
end


% Create initial value for Sigma_b
Sigma_b = 0;
for i=ind
    
    Bstar = Bv{i} - B_new;
    Sigma_b = Sigma_b + Bstar*Bstar';
    
    disp(i)
end


Bi_new = Bv;
Sigma_b_new = Sigma_b;



% Start EM-algorithm

reps = 4;
for r=1:reps
    Bi_old = Bi_new;
    B_old = B_new;
    Sigma_b_old = Sigma_b_new;

    invSig = inv(Sigma_b_new+eye(num));
    invVi = [];
    Wd = 0;

    % Pre-compute some inverses

    for i=ind    
        invVi{i} = inv(invSig + invCi{i});
        Wd = Wd + invSig;
    end


    % E-step
    Bi_new = [];
    for i=ind
        Bi_new{i} = invVi{i} * (invCi{i}*Bv{i} + invSig*Bi_old{i});
    end

    % M-step

    W = [];
    iWd = inv(Wd);
    B_new = 0;
    Sigma_b_new = 0;

    for i=ind
        W{i} = iWd*invSig;
        B_new = B_new + W{i}*Bi_new{i};
    end

    for i=ind
        Bstar = Bi_new{i}-B_new;
        Sigma_b_new = Sigma_b_new + (invVi{i} + Bstar*Bstar')/nsub;
    end
    

    disp(r)
    norm(B_new-B_old)
    
   
end

%%

Sigma_b = Sigma_b_new;
Bfinal = B_new;

Sigma = 0;

for i= ind

    Sigma = Sigma + inv(C{i} + Sigma_b);
    
end

SigmaFinal = inv(Sigma);

%%

nsub = length(ind);
c = mean(B);

Cov = [];

for i=ind
    
    name = strcat('Results_sub',num2str(i));
    load(name)
    Cov{i} = Ci + Sigma_b;
    
end


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inference

% Compute area under curve

Q = reshape(B_new,31,286)';
Q1 = Q(:,2:16)*B';
Q2 = Q(:,17:end)*B';

QQ1 = sum(Q1(:,4:12),2);
QQ2 = sum(Q2(:,4:12),2);

num = size(B,1);

u = zeros(1,num);
u(5:12) = 1;

VV1 = zeros(286,1);
VV2 = zeros(286,1);

for i=1:286, 
    VV1(i) = u*B*SigmaFinal((i-1)*31+2:(i-1)*31+16,(i-1)*31+2:(i-1)*31+16)*B'*u';
    VV2(i) = u*B*SigmaFinal((i-1)*31+17:(i)*31,(i-1)*31+17:(i)*31)*B'*u';
end


TT = (QQ2-QQ1)./(sqrt(VV1+VV2));


P = 1-tcdf(TT,25) + eps;

thr = 5.8;          % Vary threshold
ind = (TT > thr);
sum(ind)
load('resampled_mask_Buckner_r286.mat')
img = which('Buckner17subcl_SPMAnat_Morel_PAG_combined.nii');
r = region(img, 'unique_mask_values');
cluster_orthviews(r)
rr=r(ind);


PP = P(ind);

for i=1:length(rr), rr(i).Z = ones(size(rr(i).Z))*PP(i); end


%%

A1 = zeros(regions,regions);
A2 = zeros(regions,regions);

C1 = zeros(30*regions,30*regions);
C2 = zeros(30*regions,30*regions);


for i = ind,

    for j=1:286, 
        for k=1:286, 

            Si = Cov{i};
            A1(j,k) = A1(j,k) + (1/nsub)*c*Si((j-1)*31+2:(j-1)*31+16,(k-1)*31+2:(k-1)*31+16)*c';
            A2(j,k) = A2(j,k) + (1/nsub)*c*Si((j-1)*31+17:(j)*31,(k-1)*31+17:(k)*31)*c';

            C1((j-1)*30+1:(j*30),(k-1)*30+1:(k*30)) = C1((j-1)*30+1:(j*30),(k-1)*30+1:(k*30)) + (1/nsub)*B*Si((j-1)*31+2:(j-1)*31+16,(k-1)*31+2:(k-1)*31+16)*B';
            C2((j-1)*30+1:(j*30),(k-1)*30+1:(k*30)) = C2((j-1)*30+1:(j*30),(k-1)*30+1:(k*30)) + (1/nsub)*B*Si((j-1)*31+17:(j)*31,(k-1)*31+17:(k)*31)*B';

        end
    end

end

%%


tmp = diag(A1)*diag(A1)';
cA1 = A1./sqrt(tmp);

tmp = diag(A2)*diag(A2)';
cA2 = A2./sqrt(tmp);


%idx_286_for_Buck17.dat(ind == 1,3)
[a ind2] = sort(idx_286_for_Buck17.dat(ind == 1,3));
ccA1 = cA1(ind==1,ind==1);
ccA2 = cA2(ind==1,ind==1);
ttA1 = 0.5.*log((1+ccA1)./(1-ccA1));
ttA2 = 0.5.*log((1+ccA2)./(1-ccA2));


tA1 = 0.5.*log((1+cA1)./(1-cA1));
tA2 = 0.5.*log((1+cA2)./(1-cA2));
D = tA2-tA1;
DD = D(ind==1,ind==1);
imagesc(DD(ind2,ind2))

tmp = diag(C1)*diag(C1)';
cC1 = C1./sqrt(tmp);

tmp = diag(C2)*diag(C2)';
cC2 = C2./sqrt(tmp);

kind = kron(ind, ones(30,1));


ccC1 = cC1(kind==1,kind==1);
ccC2 = cC2(kind==1,kind==1);

len = length(rr);

kind2 = zeros(len,30);
for i=1:len,
    kind2(i,:) = ((ind2(i)-1)*30+1):ind2(i)*30;
end

kind2 = reshape(kind2',30*len,1);

tC1 = 0.5.*log((1+cC1)./(1-cC1));
tC2 = 0.5.*log((1+cC2)./(1-cC2));
D2 = tC2-tC1;

DD2 = D2(kind==1,kind==1);

imagesc(DD2(kind2,kind2))


M1 =  tC1(kind==1,kind==1);
M1 = M1(kind2,kind2);

M2 =  tC2(kind==1,kind==1);
M2 = M2(kind2,kind2);

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE FIGURES

cluster_orthviews(rr)
obj = fmridisplay;
obj = montage(obj, 'axial', 'slice_range', [-40 50], 'onerow', 'spacing', 8);
obj = removeblobs(obj);
obj = addblobs(obj, rr,'maxcolor', [1 1 0], 'mincolor', [1 0 0]);


rez=400; %resolution (dpi) of final graphic
f=gcf; %f is the handle of the figure you want to export
figpos=getpixelposition(f); %dont need to change anything here
resolution=get(0,'ScreenPixelsPerInch'); %dont need to change anything here
set(f,'paperunits','inches','papersize',figpos(3:4)/resolution,'paperposition',[0 0 figpos(3:4)/resolution]); %dont need to change anything here
name = 'Results072117.png'
print(f,name,'-dpng',['-r',num2str(rez)],'-opengl') %save file

%%
figure
subplot 231
imagesc(cA1,[-1 1])
axis off
title('\fontsize{16}Warm')
subplot 232
imagesc(cA2,[-1 1])
title('\fontsize{16}Hot')
axis off
subplot 233
imagesc(cA2-cA1,[-1 1])
axis off
title('\fontsize{16}Hot - Warm')

subplot 234
imagesc(ttA1(ind2,ind2),[-1 1])
% idx_286_for_Buck17.descript
axis off
title('\fontsize{16}Warm')
subplot 235
imagesc(ttA2,[-1 1])
title('\fontsize{16}Hot')
axis off
subplot 236
imagesc(DD,[-.2 .2])
axis off
colorbar
title('\fontsize{16}Hot - Warm')
colorbar

%%

figure;
subplot 131
imagesc(tC1(kind==1,kind==1),[-1 1])
axis off
title('\fontsize{16}Warm')

subplot 132
imagesc(tC2(kind==1,kind==1),[-1 1])
axis off
title('\fontsize{16}Hot')

subplot 133;
imagesc(DD2,[-1 1])
axis off
title('\fontsize{16}Hot - Warm')
colorbar


rez=400; %resolution (dpi) of final graphic
f=gcf; %f is the handle of the figure you want to export
figpos=getpixelposition(f); %dont need to change anything here
resolution=get(0,'ScreenPixelsPerInch'); %dont need to change anything here
set(f,'paperunits','inches','papersize',figpos(3:4)/resolution,'paperposition',[0 0 figpos(3:4)/resolution]); %dont need to change anything here
name = 'CorrTL072117.png';
print(f,name,'-dpng',['-r',num2str(rez)],'-opengl') %save file

figure
M1 = tC1(kind==1,kind==1);
M2 = tC2(kind==1,kind==1);
imagesc(M1,[-1 1])
size(M1)
subplot 311
imagesc(M1(61:90,:),[-1 1])
axis off
title('\fontsize{16}Warm')

subplot 312
imagesc(M2(61:90,:),[-1 1])
axis off
title('\fontsize{16}Hot')

subplot 313
imagesc(M2(61:90,:)-M1(61:90,:),[-1 1])
axis off
title('\fontsize{16}Hot - Warm')



rez=400; %resolution (dpi) of final graphic
f=gcf; %f is the handle of the figure you want to export
figpos=getpixelposition(f); %dont need to change anything here
resolution=get(0,'ScreenPixelsPerInch'); %dont need to change anything here
set(f,'paperunits','inches','papersize',figpos(3:4)/resolution,'paperposition',[0 0 figpos(3:4)/resolution]); %dont need to change anything here
name = 'CorrTLreg072117.png';
print(f,name,'-dpng',['-r',num2str(rez)],'-opengl') %save file