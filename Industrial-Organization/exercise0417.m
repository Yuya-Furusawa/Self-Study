% 2019/04/17
% Exercise in IO class
% Estimate the following equation
% y_i = \beta_0 + \beta_1^{x1_i} + \beta_2 * x2 + \epsilon


DATA = load('DataProblem201904.csv');  %Date, size=(1000,3)
y = DATA(:,1);
n = size(DATA, 1);
x0 = ones(n,1);
x1 = DATA(:,2);
x2 = DATA(:,3);

m = size(DATA,2);
theta = ones(m ,1);
f = @nlobj;
fminsearch(f, theta)

function objvalue = nlobj(theta)
global x0 x1 x2 y n;

sum = 0;
for i = 1:n
   sum = sum + (y(i) - theta(1)*x0(i) - theta(2)^x1(i) - theta(3)*x2(i))^2;
end
objvalue = sum;
end

% ans = 1, 2, 3