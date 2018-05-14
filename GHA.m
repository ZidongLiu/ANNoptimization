function [w_final,w_ini,w_learn,Err_train,n_steps,exitflag] = GHA(X,w_range,alpha,max_ite,freq,tol)
    [~,n] = size(X);
    a = w_range(1);
    b = w_range(2);
    w_ini = rand(n,n)*(b-a)-a;
    w_learn = cell(1,max_ite/freq);
    Err_train = zeros(1,max_ite/freq);
    w_current = w_ini;
    exitflag = 0;
    w_learn = cell(1,floor(max_ite/freq));
    for i=1:max_ite
        x_online = datasample(X,1);
        y_online = w_current * x_online';
        M_decay = decay(y_online);
        w_new = w_current + alpha(i)*y_online*x_online - alpha(i) * diag(y_online)*M_decay*w_current;
        w_current = w_new;
        my_inner = w_new*w_new';
        Err = sum(sum((my_inner-eye(n)).^2));
        if(mod(i,freq)==0)
            w_learn{i/freq} = w_current;
            Err_train(i/freq) = Err;
        end
        if(Err < tol)
            exitflag = 1;
            break;
        end
    end
    w_final = w_current;
    n_steps = i;
end