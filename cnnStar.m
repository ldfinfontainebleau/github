clear all; close all; clc;  

txtlist=dir(fullfile('D:','cnn','7802','star','*.txt'));%��ȡtxt��
imglist=dir(fullfile('D:','cnn','7802','FITS','*.fit'));%��ȡfits
txtfit=cell(450,3);%ϸ������洢��һ����txt���ڶ����Ƕ�Ӧͼ�񡣵�������ÿһ��ͼ���trainsample
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
};  %5������
  
% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������  
cnn = cnnsetup(cnn, train_x, train_y);  
  
% ѧϰ��  
opts.alpha = 1;  
% ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������  
% �����������������ˣ�������������������˲ŵ���һ��Ȩֵ  
opts.batchsize = 50;   
% ѵ����������ͬ��������������ѵ����ʱ��  
% 1��ʱ�� 11.41% error  
% 5��ʱ�� 4.2% error  
% 10��ʱ�� 2.73% error  
opts.numepochs = 10;  
  
% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����  
cnn = cnntrain(cnn, train_x, train_y, opts);  
  
% Ȼ����ò�������������  
[er, bad] = cnntest(cnn, test_x, test_y); %����һ��ͼ 
  
%plot mean squared error  
plot(cnn.rL);  
%show test error  
disp([num2str(er*100) '% error']);