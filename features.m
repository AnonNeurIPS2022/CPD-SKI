function Mati = features(X,U,lengthscale)
    Mati = SE(X,linspace(0,1,size(U,1))',lengthscale)/U;
end