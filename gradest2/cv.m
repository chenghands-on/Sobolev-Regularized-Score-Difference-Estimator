function avg_loss = cv(med, xp, xq)
    % Inputs:
    % loss_func - Function handle for computing validation loss
    % data - Dataset, a matrix where rows are observations and columns are features

    % Output:
    % avg_loss - Average validation loss from cross-validation

    % Number of folds
    k = 5;

    % Determine the size of each fold
    [n, ~] = size(xp);
    fold_size = floor(n / k);

    % Initialize total loss
    total_loss = 0;

    % Perform k-fold cross-validation
    for i = 1:k
        
        % Determine indices for validation and training data
        val_indices = (i-1)*fold_size + 1 : i*fold_size;
        train_indices = setdiff(1:n, val_indices);

        % Split the data into training and validation sets
        xp_tr = xp(train_indices, :); xq_tr = xq(train_indices, :);
        xp_te = xp(val_indices, :); xq_te = xq(val_indices, :);

        [xp_tr, xq_tr] = upload(xp_tr, xq_tr);
        [W_tr,b_tr] = GradEst_fdiv(xp_tr, xq_tr, [xp_te; xq_te], @(d) exp(d - 1), med, 2000);
        [W_tr, b_tr] = download(W_tr, b_tr);

        dpt = sum(W_tr(1:size(xp_te, 1), :).*xp_te, 2) + b_tr(1:size(xp_te, 1));
        dqt = sum(W_tr(size(xp_te, 1)+1:end, :).*xq_te, 2) + b_tr(size(xp_te, 1)+1:end);

        testing_loss = -mean(dpt) + mean(exp(dqt-1));

        % Compute the validation loss and add to total loss
        total_loss = total_loss + testing_loss;
    end

    % Compute average loss
    avg_loss = total_loss / k;
end

function [xp, xq] = upload(xp, xq)
    xp = gpuArray(single(xp)); 
    xq = gpuArray(single(xq));
end

function [w, b] = download(w, b)
    w = double(gather(w));
    b = double(gather(b));
end