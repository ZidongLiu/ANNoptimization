function w_learn = SOM_learn(X,shape,max_ite,freq,alpha,eta_0)
    [m,n] = size(X);
    upper = max(X);
    lower = min(X);
    a = upper-lower;
    b = (upper+lower)/2;
    %X_stand = X * diag(1./upper);
    w = cell(shape);
    dist_mat = zeros(shape);
    w_learn = cell(1,1);
    learn_ind = 1;
    for i=1:shape(1)
        for j= 1:shape(2)
            w{i,j} = a.*rand(1,n)+b;
        end
    end
    for t = 1:max_ite
        % selection
        x = X(randsample(m,1),:);
        % 1.comparison
        for j =1:shape(1)
            for k =1:shape(2)
                dist_mat(j,k) = norm(w{j,k}-x);
            end
        end
        [~,I] = min(dist_mat(:));
        [I_row,I_col] = ind2sub(shape,I);
        c = [I_row,I_col];
        % update
         for j =1:shape(1)
            for k =1:shape(2)
                h_jk = h(c,j,k,eta_0,t,max_ite/2);
                w{j,k} = w{j,k} + alpha(t)*h_jk*(x-w{j,k});
            end
         end
        ind = sum(freq(2,:)<t);
        res = t - freq(2,ind);
        div = freq(1,ind+1);
        if mod(res,div)==0
            w_learn{1,learn_ind}.w = w;
            w_learn{1,learn_ind}.step = t;
            learn_ind = learn_ind+1;
        end
    end
end