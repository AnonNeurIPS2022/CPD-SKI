function error = approximationError(X,U,lengthscale)
    perm = randperm(min(1000,size(X,1)));
    XPerm = X(perm,:);
    K = exp(-0.5*pdist2(XPerm,XPerm).^2/lengthscale^2);
    KApprox = ones(size(K));
    for d = 1:size(XPerm,2)
        phi = features(XPerm(:,d),U,lengthscale);
        KApprox = KApprox.*(phi*phi');
    end
    error = norm(K-KApprox,'fro')/norm(K,'fro');
end
