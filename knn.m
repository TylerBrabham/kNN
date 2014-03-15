function [accuracy] = knn(metric)

if nargin<1
	metric = 'euc';
end

[X_set,y_set] = load_train();

X_train = X_set{4};
y_train = y_set{4};

%open test data
[X_test, y_test] = load_test();

predictions = predict(X_train,y_train,X_test, metric);

%display(predictions);

accuracy = sum(predictions==y_test) / length(y_test);

end % end main function

%%%%%%%%%%%%%%%%%%%%
% Cross Validation %
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%
% Testing %
%%%%%%%%%%%
function [accuracy] = compute_accuracy(y, predictions)



end %end subfunction

function [predictions] = predict(X_train,y_train,X_test, metric)

if nargin<4
	metric = 'euc';
end

alpha = .005;

[n,d] = size(X_train);
[m,d] = size(X_test);

predictions = -1 * ones(m,1);

less_than = @(a,b) a < b;

greater_than = @(a,b) a > b;

%set distance func, and comparison method
if strcmp(metric,'euc')
	distance_func = @(x,y, digit) norm(x-y);

	compar_func = less_than;
	extreme_dist = inf;
elseif strcmp(metric, 'man')
	distance_func = @(x,y, digit) sum(abs(x-y));

	compar_func = less_than;
	extreme_dist = inf;
elseif strcmp(metric, 'mahal')
	S_cell = cell(10,1);
	for digit=0:9
		sigma = cov(X_train(y_train==digit));

		sigma = sigma + alpha * eye(d);

		S_inv = inv(sigma);

		S_cell{digit+1} = S_inv;
	end

	distance_func = @(x,y, digit) sqrt((x-y) * S_cell{digit+1} * (x-y)');

	compar_func = less_than;
	extreme_dist = inf;
elseif strcmp(metric, 'cos')
	distance_func = @cosine_metric;

	compar_func = greater_than;
	extreme_dist = -inf;
end

%scan through data
for i=1:m
	test_vec = X_test(i,:);

	best_dist = extreme_dist;

	label = -1;
	for j=1:n
		train_vec = X_train(j,:);
		cur_dist = distance_func(train_vec,test_vec, y_train(j));

		if compar_func(cur_dist, best_dist)
			best_dist = cur_dist;
			label = y_train(j);
		end
	end

	display(i);

	predictions(i) = label;
end

end %end subfunction

%%%%%%%%%%%%%%%%%%
% Math Functions %
%%%%%%%%%%%%%%%%%%
function dist = cosine_metric(x,y, digit)

dist = dot(x,y);

% if dot(x,x)==0 || dot(y,y)==0
% 	dist = 0;
% else

% 	dist = dot(x,y) / sqrt(dot(x,x) * dot(y,y));
% end

end % end subfunction

%%%%%%%%%%%%%%%%%
% I/O Functions %
%%%%%%%%%%%%%%%%%
function [X_set, y_set] = load_train()
%Load all seven data sets and labels into a cell. Flatten matrix and normalize vectors.

train_data = open('train_small.mat');

X_set = cell(7,1);
y_set = cell(7,1);

for i = 1:7
	X_set{i} = flatten_3dmatrix(train_data.train{i}.images);
	y_set{i} = train_data.train{i}.labels;
end

end %end subfunction

function [X_train, y_train] = load_full_train()
%Load full train data

train_data = open('train.mat');

X_train = flatten_3dmatrix(train_data.train.images);
y_train = train_data.train.labels;

end %end subfunction

function [X_test,y_test] = load_test()
%Load test data sets and labels. Flatten matrix and normalize vectors.
test_data = open('test.mat');
X_test = flatten_3dmatrix(test_data.test.images);
y_test = test_data.test.labels;

end

function [X_train] = flatten_3dmatrix(x_arg)
%Simple helper method for flattening a 3d matrix by 
%concatenating all rows of matrix into a single row.

[m,n,train_size] = size(x_arg);

X_train = zeros(train_size, m*n);

for i = 1:train_size
	X_train(i,:) = reshape(double(x_arg(:,:,i)),1,[]);
	X_train(i,:) = X_train(i,:)/norm(X_train(i,:));
end

end %end subfunction