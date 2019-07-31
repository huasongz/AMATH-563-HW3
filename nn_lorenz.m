clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; 

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  x(4) * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3);
                  0]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

figure(1)
input1=[]; output1=[];
for j=1:100  % training trajectories
    x0=[30*(rand(3,1)-0.5);10];
    [t,y] = ode45(Lorenz,t,x0);
    input1=[input1; y(1:end-1,:)];
    output1=[output1; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
xlabel('x')
ylabel('y')
zlabel('z')
grid on, view(-23,18)

figure(2)
input2=[]; output2=[];
for j=1:100  % training trajectories
    x0=[30*(rand(3,1)-0.5);28];
    [t,y] = ode45(Lorenz,t,x0);
    input2=[input2; y(1:end-1,:)];
    output2=[output2; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
xlabel('x')
ylabel('y')
zlabel('z')
grid on, view(-23,18)

figure(3)
input3=[]; output3=[];
for j=1:100  % training trajectories
    x0=[30*(rand(3,1)-0.5);40];
    [t,y] = ode45(Lorenz,t,x0);
    input3=[input3; y(1:end-1,:)];
    output3=[output3; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
xlabel('x')
ylabel('y')
zlabel('z')
grid on, view(-23,18)

%%
n = randi([1 8000],1,100);
input = [input1(n,:);input2(n,:);input3(n,:)];
output = [output1(n,:);output2(n,:);output3(n,:)];

%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%% p = 17
figure(6)
x0=[20*(rand(3,1)-0.5);17];
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on % ground truth
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2]) % initial value
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('true trajectory','initial value','prediction')
title('rho = 17')

figure(7)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title('x direction')
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
title('y direction')
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
title('z direction')

% p = 35
figure(8)
x0=[20*(rand(3,1)-0.5);35];
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
legend('true trajectory','initial value','prediction')
title('rho = 35')

figure(7)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title('x direction')
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
title('y direction')
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
title('z direction')
%suptitle('comparison between prediction and true trajectory')

figure(6), view(-75,15)
figure(8), view(-75,15)
figure(7)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Lorenz','NN')

%%
r = 28;
inputx = [];outputx = [];
inputy = [];outputy = [];
%for j=1:100  % training trajectories
    x0=[30*(rand(3,1)-0.5);r];
    [t,y] = ode45(Lorenz,t,x0);
    inputx=[inputx; y(1:end-1,1)];
    inputy = [inputy; y(1:end-1,2)];
    outputx=[outputx; y(2:end,1)];
    outputy = [outputy; y(2:end,2)];
    %plot3(y(:,1),y(:,2),y(:,3)), hold on
    %plot3(x0(1),x0(2),x0(3),'ro')
    %xlabel('x')
    %ylabel('y')
    %zlabel('z')
%end

plot(inputx,inputy)
hold on
plot(-20:20,2.*[-20:20],'LineWidth',[2])
xlabel('x')
ylabel('y')


lobe = [];
for i = 1:length(inputx)
    if inputy(i) > 2*inputx(i)
        lobe = [lobe;1];
    else 
        lobe = [lobe;-1];
    end
    
%     if output(i)*output(i+1) < 0
%         output(i+1) = -1;
%     else
%         output(i+1) = 1;
%     end
end

label = [1];
a = 0;
for i = length(lobe):-1:2
    if sign(lobe(i)*lobe(i-1)) == -1
        a = 0;
        label = [a;label]; % if jumps to another lobe, denote as 1
    else
        a = a+1;
        label = [a;label]; % if stays, denote as 0
    end
        
end
figure()
plot(label)
training = [outputx(1:length(label)) outputy(1:length(label))];

%%
net1 = feedforwardnet([10 10 10]);
net1.layers{1}.transferFcn = 'logsig';
net1.layers{2}.transferFcn = 'radbas';
net1.layers{3}.transferFcn = 'purelin';
net1 = train(net1,training.',label.');

%% test NN
x0=[30*(rand(3,1)-0.5);28];
[t,y] = ode45(Lorenz,t,x0);
performance = net1(y(:,1:2).');

lobe1 = [];
for i = 1:length(y(:,1))
    if y(i,2) > 2*y(i,1)
        lobe1 = [lobe1;1];
    else 
        lobe1 = [lobe1;-1];
    end
end

label1 = [];
a = 0;
for i = length(lobe1):-1:2
    if sign(lobe1(i)*lobe1(i-1)) == -1
        a = 0;
        label1 = [a;label1]; % if jumps to another lobe, denote as 1
    else
        a = a+1;
        label1 = [a;label1]; % if stays, denote as 0
    end
        
end

plot(performance,'r','LineWidth',[2])
hold on
plot(label1,'b','LineWidth',[2])
legend('prediction','true')
title('prediction of jumping between lobes')




        

