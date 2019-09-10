clc;
fprintf(['|For the data set ' CurData '\n']);

if isfield(options,'lambda')
    Rec = SRC_rec;
    time = SRC_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of SRC is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf(' * when lambda = %f and auto Dim = %d\n\n', options.lambda(ind), options.k(ind) );


    Rec = SVM_rec;
    time = SVM_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of SVM is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf(' * when lambda = %f and auto Dim = %d\n\n', options.lambda(ind), options.k(ind) );

    Rec = KNN_rec;
    time = KNN_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of NN is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf(' * when lambda = %f and auto Dim = %d\n\n', options.lambda(ind), options.k(ind) );

else
    Rec = SRC_rec;
    time = SRC_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of SRC is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf([' * when k = ' num2str(options.k(ind)) '\n'])

  
    Rec = SVM_rec;
    time = SVM_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of SVM is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf([' * when k = ' num2str(options.k(ind)) '\n'])

    Rec = KNN_rec;
    time = KNN_time;
    [val ind] = max(Rec);
    fprintf(' | The classification accuracy of NN is about %4.2f and the time cost is about %4.2f seconds\n', val*100, time(ind));
    fprintf([' * when k = ' num2str(options.k(ind)) '\n'])

end

