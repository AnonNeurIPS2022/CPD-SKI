function [W, U] = CPLS(X, y, M, R,lambda, lengthscale, numberSweeps)
[~, D] = size(X);
W = cell(1,D);
Matd = 1;
reg = 1;
w = linspace(0,1,M)';
U = SE(w,w,lengthscale);
[~,S,U] = svd(U);
U = sqrt(S)*U';
    for d = D:-1:1
        W{d} = randn(M,R);
        W{d} = W{d}/norm(W{d});
        reg = reg.*(W{d}'*W{d});
        Mati = features(X(:,d),U,lengthscale);
        Matd = (Mati*W{d}).*Matd;
    end
    itemax = numberSweeps*(2*(D-1))+1;
    for ite = 1:itemax
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        Mati = features(X(:,d),U,lengthscale);
        reg = reg./(W{d}'*W{d});
        Matd = Matd./(Mati*W{d});
        [CC,Cy] = dotkronLargeScale(Mati,Matd,y);
        x = (CC+lambda*kron(reg,eye(M)))\Cy;
        clear CC Cy
        W{d} = reshape(x,M,R);
        reg = reg.*(W{d}'*W{d});
        Matd = Matd.*(Mati*W{d});
    end
end