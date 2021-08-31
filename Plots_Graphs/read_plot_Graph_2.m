clc;
%close all;
clear

% here I am plotting the training accuracies of word and patch images
% together

textFilePathSubSeq1 = '/home/tmondal/Documents/Project_Work/Python_Project/Font_Recognition/OAR.word_multitask_1_short_plus.stdout';
textFilePathSubSeq2 = '/home/tmondal/Documents/Project_Work/Python_Project/Font_Recognition/OAR.patch_multi_tasks.stdout';

[keepAllTrainAccuWord, keepAllValidAccuWord] = ReadFile_1(textFilePathSubSeq1);
[keepAllTrainAccuPatch, keepAllValidAccuPatch] = ReadFile_2(textFilePathSubSeq2);




keepAllTrainAccuWord(:,5) = 1:length(keepAllTrainAccuWord);
keepAllTrainAccuPatch(:,5) = 1:length(keepAllTrainAccuPatch);

figure();
h(1) = subplot(1,1,1);

colorspec = {[0.9 0.9 0.9]; [0.8 0.8 0.8]; [0.6 0.6 0.6]; ...
    [0.4 0.4 0.4]; [0.2 0.2 0.2];[0 0.75 0.75];[0 0.5 0];[0.75 0.75 0];...
    [1 0.50 0.25];[0.75 0 0.75];[0.7 0.7 0.7];[0.8 0.7 0.6];[0.6 0.5 0.4 ]};



%% Plot for the protien and sheep data

q = plot(keepAllTrainAccuWord(:,5),keepAllTrainAccuWord(:,1),'r-*', ...
    keepAllTrainAccuPatch(:,5),keepAllTrainAccuPatch(:,1),'r--d', ...
    keepAllTrainAccuWord(:,5),keepAllTrainAccuWord(:,2),'b-*',...
    keepAllTrainAccuPatch(:,5),keepAllTrainAccuPatch(:,2),'b--d',...
    keepAllTrainAccuWord(:,5),keepAllTrainAccuWord(:,3),'m-*',...
    keepAllTrainAccuPatch(:,5),keepAllTrainAccuPatch(:,3),'m--d',...
    keepAllTrainAccuWord(:,5),keepAllTrainAccuWord(:,4),'k-*',...
    keepAllTrainAccuPatch(:,5),keepAllTrainAccuPatch(:,4),'k--d');

hold on;
hleg1 = legend('Word: Scanning Accuracy', 'Patch: Scanning Accuracy', ...
    'Word: Font Size Accuracy', 'Patch: Font Size Accuracy', 'Word: Font Type Accuracy', 'Patch: Font Type Accuracy', ...
    'Word: Font Emphasis Accuracy', 'Patch: Font Emphasis Accuracy');

set(hleg1,'Location','NorthWest')
set(hleg1,'FontSize',14)
set(gca,'XTick', keepAllTrainAccuWord(:,5));
grid on;
set(gca,'FontSize',10);
xl = xlabel('Number of Epochs');
yl = ylabel('Accuracy');
set(xl,'FontSize',14,'FontWeight','bold','FontName','Courier');
set(yl,'FontSize',14,'FontWeight','bold','FontName','Courier');
hold off;
disp('see me');




function [keepAllTrainAccu, keepAllValidAccu] = ReadFile_1(textFilePath)

fid = fopen(textFilePath);
tline = fgetl(fid);

keepAllTrainAccu = zeros(1,4);
trainCnt = 1;

keepAllValidAccu = zeros(1,4);
validCnt = 1;

while ischar(tline)
    newStr = split(tline,' ');
    if( (strcmp(newStr{1}, 'train'))  && (strcmp(newStr{2}, 'scanning_Acc:')))
        keepAllTrainAccu(trainCnt, 1) = str2double(newStr{3});
        keepAllTrainAccu(trainCnt, 2) = str2double(newStr{7});
        keepAllTrainAccu(trainCnt, 3) = str2double(newStr{11});
        keepAllTrainAccu(trainCnt, 4) = str2double(newStr{14});
        trainCnt = trainCnt +1;
    end
    
    if( (strcmp(newStr{1}, 'val')) && (strcmp(newStr{2}, 'scanning_Acc:')) )
        keepAllValidAccu(validCnt, 1) = str2double(newStr{3});
        keepAllValidAccu(validCnt, 2) = str2double(newStr{7});
        keepAllValidAccu(validCnt, 3) = str2double(newStr{11});
        keepAllValidAccu(validCnt, 4) = str2double(newStr{14});
        validCnt = validCnt +1;
    end
    tline = fgetl(fid);
end

fclose(fid);
return;
end

function [keepAllTrainAccu, keepAllValidAccu] = ReadFile_2(textFilePath)
fid = fopen(textFilePath);
tline = fgetl(fid);

keepAllTrainAccu = zeros(1,4);
trainCnt = 1;

keepAllValidAccu = zeros(1,4);
validCnt = 1;

while ischar(tline)
    newStr = split(tline,' ');
    if( (strcmp(newStr{1}, 'train'))  && (strcmp(newStr{2}, 'scanning_Acc:')))
        keepAllTrainAccu(trainCnt, 1) = str2double(newStr{3});
        keepAllTrainAccu(trainCnt, 2) = str2double(newStr{7});
        keepAllTrainAccu(trainCnt, 3) = str2double(newStr{11});
        keepAllTrainAccu(trainCnt, 4) = str2double(newStr{14});
        trainCnt = trainCnt +1;
    end
    
    if( (strcmp(newStr{1}, 'val')) && (strcmp(newStr{2}, 'scanning_Acc:')) )
        keepAllValidAccu(validCnt, 1) = str2double(newStr{3});
        keepAllValidAccu(validCnt, 2) = str2double(newStr{7});
        keepAllValidAccu(validCnt, 3) = str2double(newStr{11});
        keepAllValidAccu(validCnt, 4) = str2double(newStr{14});
        validCnt = validCnt +1;
    end
    tline = fgetl(fid);
end

fclose(fid);
return;
end

