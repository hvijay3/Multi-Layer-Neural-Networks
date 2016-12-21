function [e2] = compute_error(e3,Parameter2,A2)
e2 = Parameter2'*e3.*(A2.*(1- A2));
end