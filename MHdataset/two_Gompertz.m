function [ y ] = two_Gompertz( x )
a1=0.5247;
b1=6;
c1=1.2;

a=0.4;
b=6;
c=2.5;

 for i=1:size(x,1)
    for j=1:size(x,2)
        if x(i,j)<0.35
             y(i,j)=a1*exp(-b1*exp(-c1*(11*(x(i,j)+0.6-0.13)-5)));
        else
             y(i,j)=a*exp(-b*exp(-c*(11*(x(i,j)-0.13)-5)))+0.5;
        end
    end
end
% plot(x,y);grid on
end
