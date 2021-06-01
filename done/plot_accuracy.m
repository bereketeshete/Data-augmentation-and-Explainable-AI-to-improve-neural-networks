
% plot accuracy vs iteration(epoch)

model_dir='trained_model_audio_max_180618180809';
files = dir(sprintf('%s/*.mat',model_dir));

for i=1:length(files)
    b{i}=files(i).name;
    n=strsplit(b{i},'_');
    p=strsplit(char(n(6)),'.');
    A(i,1)=str2num(n{2});
    A(i,2)=str2num(p{1});
    A=sortrows(A);
end 


epoch=A(:,1);
accuracy=A(:,2);

plot(epoch,accuracy)
title('Accuracy vs epoch');
xlabel('epoch (sampled every 1000 iterations)');
ylabel('Accuracy');
xlim([A(1,1) A(length(files),1)])
ylim([0 100])