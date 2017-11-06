function [M_est U_est V_est] = RMC_Lp(M,W,r,rho,maxIterIN,p)
%% Robust low-rank matrix approximation with missing data and outliers
% min |W.*(M-E)|_p
% s.t., E = UV, U'*U = I
%
%Input: 
%   M: m*n data matrix
%   W: m*n indicator matrix, with '1' means 'observed', and '0' 'missing'.
%   r: the rank of r
%   lambda: the weighting factor of the trace-norm regularization, 1e-3 in default.
%   rho: increasing ratio of the penalty parameter mu, usually 1.05.
%  maxIterIN£º 1 in default.
%   p: p norm;
%Output:
%   M_est: m*n full matrix, such that M_est = U_est*V_est
%   U_est: m*r matrix
%   V_est: r*n matrix
%% Normalization
scale = max(max(abs(M)));
M = M/scale;
M = M.*W;
%% In-default parameters
[m n] = size(M); %matrix dimension

maxIterOUT = 500;
max_mu = 1e20;
mu = 1e-6;        
M_norm = norm(M,'fro');
tol = 1e-8*M_norm;
Omega = find(W);
cW = ones(size(W)) - W; %the complement of W.
%% Initializing optimization variables 
E = zeros(m,n);
U = ones(m,r);
V = ones(r,n);
Y = zeros(m,n); %lagrange multiplier
%% Start main outer loop
iter_OUT = 0;
objs=[];
while iter_OUT < maxIterOUT
    iter_OUT = iter_OUT + 1;
    
    itr_IN = 0;
    obj_pre = 1e20;                          %
    %start inner loop
    while itr_IN < maxIterIN 
        
        %update U
         temp = (E + Y/mu)*V';
         [U,~] = qr(temp,0);
        %% update V 

        V = U'*(E + Y/mu);
        UV = U*V;
        
       %% update E
        temp1 = UV - Y/mu;
        Lt = M-E;
        temp = M-UV+Y/mu;
        for i = 1:length(Omega)
            Lt(Omega(i)) = find_root_p(temp(Omega(i)), 1/mu,p);
        end
        Lt = Lt.*W+temp.*cW;
        E = M-Lt;

        %evaluate current objective
        obj_cur = sum(sum(abs(W.*(M-E)))) + sum(sum(Y.*(E-UV))) + mu/2*norm(E-UV,'fro')^2;

        %check convergence of inner loop %  
        if abs(obj_cur - obj_pre) <= 1e-8*abs(obj_pre)
            break;
        else
            obj_pre = obj_cur;
            itr_IN = itr_IN + 1;
        end
    end
    leq = E - UV;
    stopC = norm(leq,'fro');
    if display
        obj = sum(sum(abs(W.*(M-UV)))); 
        objs = [objs,obj];
    end
    if display && (iter_OUT==1 || mod(iter_OUT,50)==0 || stopC<tol)
        disp(['iter ' num2str(iter_OUT) ',mu=' num2str(mu,'%2.1e') ...
            ',obj=' num2str(obj) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        %update lagrage multiplier
        Y = Y + mu*leq;
        %update penalty parameter
        mu = min(max_mu,mu*rho);
    end
end

%% Denormalization
U_est = sqrt(scale)*U; V_est = sqrt(scale)*V;
M_est = U_est*V_est;
end
%% 
function x = find_root_p(alpha, lambda,p)

const1 = (2*lambda*(1-p))^(1/(2-p));
const2 = const1 + lambda*p*const1^(p-1);

if abs(alpha)<const2
    x = 0;
elseif abs(alpha) == const2
    obj1 = 0.5*alpha^2;
    temp = sign(alpha)*const1;
    obj2 = 0.5*(temp-alpha)^2+lambda*abs(temp)^p;
    if obj1 <obj2
    x = 0;
    else
    x = temp;
    end
else
    x = const1;
    for i = 1:2
     x = abs(alpha) - lambda*p*x^(p-1);  
    end
    x = sign(alpha)*x;
end
end