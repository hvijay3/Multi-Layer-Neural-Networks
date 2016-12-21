function C = cost(H,Y)
sum = 0 



end
function [ C ] = cost(e )


M = size(Xtrain,1);  %Number of training examples

oldsum = 0;

for i = 1:M
    [A1,A2,A3,H] = forwardprop(A1,L1,L2,L3,theta);
        H = forwardprop(Xtrain,W1,W2); 
        temp = ( H(i) - Ytrain(i) )^2;
        Sum = temp + oldsum;
        oldsum = Sum;
end

C = (1/2*M) * Sum;

end