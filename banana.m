%% Banana Dataset Plots
M = 50;
RSet = [2,4,6,M];
lambda = 1e-6;
lengthscale = 0.5;
NIte = 10;
%% Generate Plots
close all
rng('default');
warning off
X = readmatrix('banana.csv');
N = size(X,1);
X = X(randperm(size(X,1)),:);
Y = (X(:,end)==1)-(X(:,end)==2);
X = X(:,1:2);
XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

NPlot = 100;
X1Plot = linspace(0,1,NPlot);
[X1Plot,X2Plot] = meshgrid(X1Plot,X1Plot);
XPlot = [X1Plot(:),X2Plot(:)];
Z = ones(size(X,1),1);
ZPlot = ones(size(XPlot,1),1);
w = linspace(0,1,M)';
U = SE(w,w,lengthscale);
[~,S,U] = svd(U);
U = sqrt(S)*U';

[X1II, X2II] = meshgrid(w',w');
for d = 1:size(X,2)
    Z = dotkron(Z,features(X(:,d),U,lengthscale));
    ZPlot = dotkron(ZPlot,features(XPlot(:,d),U,lengthscale));
end
K = exp(-0.5*pdist2(X,X).^2/lengthscale^2);
wKRR = (K+lambda*eye(N))\Y;
scorePlotKRR = sign(exp(-0.5*pdist2(XPlot,X).^2/lengthscale^2)*wKRR);
discretizationError = norm(K-Z*Z')/norm(K);
disp("Discretization error: "+string(discretizationError));

%% Plots
plotIdx = 0;
for R = RSet
    rng(plotIdx);
    plotIdx = plotIdx+1;tic;
    W = CPLS(X,Y,M,R,lambda,lengthscale,NIte);toc;
    scorePlotCP = sign(CPPredict(XPlot,W,U,lengthscale));
    
    figure(plotIdx);
    fig = gcf;
    hold on
    s3 = scatter(X1II,X2II,10,'black','x');
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[216, 27, 96]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[30, 136, 229]/255,'filled');
    s1.MarkerFaceAlpha = 0.22;
    s2.MarkerFaceAlpha = 0.22;
    c3 = contour(X1Plot,X2Plot,reshape(scorePlotKRR,size(X1Plot)),[0 0],'black','LineWidth',2,'LineStyle','--');
    c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(X1Plot)),[0 0],'black','LineWidth',2.5);
    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    axis equal
    axis off
    hold off
    filename = 'banana'+string(M^2)+'inducingPoints'+string(R)+'rank'+'.pdf';
%     exportgraphics(fig,filename,'BackgroundColor','none','ContentType','vector');
end