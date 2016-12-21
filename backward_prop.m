function [e2] = backward_prop(e3,Parameter2,A2)
e2 = Parameter2'*e3.*(A2.*(1- A2));
end