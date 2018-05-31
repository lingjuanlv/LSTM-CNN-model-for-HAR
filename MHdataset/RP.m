function output = RP( input )
dim=size(input,2);
reduced_dim=floor(dim/2);
%%
%RP:row-ortho noise matrix, distance is not distorted too much
A=normrnd(0,1,[dim,reduced_dim]);
[Q, R] = qr(A); 
T=Q(:,1:size(Q,2)/2);
output=input*T; 
end
