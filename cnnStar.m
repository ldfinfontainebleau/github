clear all; close all; clc;  

txtlist=dir(fullfile('D:','cnn','7802','star','*.txt'));%读取txt。
imglist=dir(fullfile('D:','cnn','7802','FITS','*.fit'));%读取fits
txtfit=cell(450,3);%细胞数组存储第一列是txt，第二列是对应图像。第三列是每一个图像的trainsample
img_num=length(imglist);
for j=1:img_num
    txt_name=txtlist(j).name;
    txtdata=importdata(fullfile('D:','cnn','7802','star',txt_name));
    image_name=imglist(j).name;
    image=fitsread(fullfile('D:','cnn','7802','FITS',image_name));
    txtfit{j,1}=txtdata;
    txtfit{j,2}=image;
end
trainsample=cell(200,1);
for i=1:450
    numtxt=length(txtfit{i,1});
    for j=1:numtxt
        a=txtfit{i,1}(j,1);
        b=txtfit{i,1}(j,2);
        tiquimage=txtfit{i,2}(a-3:a+3,b-3:b+3);
        trainsample{j,1}=tiquimage;
    end
    txtfit{i,3}=trainsample;
end

trainx=txtfit{:,3}(:,:);
trainy=1;
%% ex1   
%will run 1 epoch in about 200 second and get around 11% error.   
%With 100 epochs you'll get around 1.2% error  
  
cnn.layers = {  
    struct('type', 'i') %input layer  
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer  
    struct('type', 's', 'scale', 2) %sub sampling layer  
    struct('type', 'c', 'outputmaps', 4, 'kernelsize', 5) %convolution layer  
    struct('type', 's', 'scale', 2) %subsampling layer  
};  %5曾网络
  
% 这里把cnn的设置给cnnsetup，它会据此构建一个完整的CNN网络，并返回  
cnn = cnnsetup(cnn, train_x, train_y);  
  
% 学习率  
opts.alpha = 1;  
% 每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是  
% 把所有样本都输入了，计算所有样本的误差了才调整一次权值  
opts.batchsize = 50;   
% 训练次数，用同样的样本集。我训练的时候：  
% 1的时候 11.41% error  
% 5的时候 4.2% error  
% 10的时候 2.73% error  
opts.numepochs = 10;  
  
% 然后开始把训练样本给它，开始训练这个CNN网络  
cnn = cnntrain(cnn, train_x, train_y, opts);  
  
% 然后就用测试样本来测试  
[er, bad] = cnntest(cnn, test_x, test_y); %用另一幅图 
  
%plot mean squared error  
plot(cnn.rL);  
%show test error  
disp([num2str(er*100) '% error']);