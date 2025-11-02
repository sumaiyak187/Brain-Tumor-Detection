clc; clear; close all;

% === Step 1: Load Image ===
imagePath = 'images\brain17.jpg';  % <-- Replace with your image path
img1 = imread(imagePath);
img1 = im2double(img1);
if size(img1,3) == 3
    img1 = rgb2gray(img1);
end
original = img1;

% === Step 2: Median Filtering ===
K = medfilt2(original);

% === Step 3: Edge Detection using Sobel Filter ===
C = double(K);
B = zeros(size(C));
for i = 1:size(C,1)-2
    for j = 1:size(C,2)-2
        Gx = (2*C(i+2,j+1) + C(i+2,j) + C(i+2,j+2)) - (2*C(i,j+1) + C(i,j) + C(i,j+2));
        Gy = (2*C(i+1,j+2) + C(i,j+2) + C(i+2,j+2)) - (2*C(i+1,j) + C(i,j) + C(i+2,j));
        B(i,j) = sqrt(Gx.^2 + Gy.^2);
    end
end

% === Step 4: Binarization ===
bw = imbinarize(K, 0.7);

% === Step 5: Connected Components ===
label = bwlabel(bw);
RGB_label = label2rgb(label, 'lines', 'k', 'shuffle');


% === Step 6: Region Property Analysis and Heuristic Filtering ===
stats = regionprops(label, 'Solidity', 'Area');
density  = [stats.Solidity];
area = [stats.Area];

high_dense_area = density > 0.5;
filtered_labels = ismember(label, find(high_dense_area));

% === Step 7: Largest Region Selection ===
max_area = max(area(high_dense_area));
tumor_label = find(area == max_area);
tumor = ismember(label, tumor_label);

% === Step 8: Morphological Dilation ===
se = strel('square', 5);
tumor_dilated = imdilate(tumor, se);

% === Step 9: Final Tumor Detection ===
Bound = bwboundaries(tumor_dilated, 'noholes');

% === Display All Stages in One Figure ===
figure('Name','Brain Tumor Detection Pipeline','NumberTitle','off');

subplot(3,4,1), imshow(original), title('\fontsize{10}Original MRI');
subplot(3,4,2), imshow(K), title('\fontsize{10}Median Filtered');
subplot(3,4,3), imshow(B, []), title('\fontsize{10}Edge Detection');
subplot(3,4,4), imshow(bw), title('\fontsize{10}Binarized Image');

subplot(3,4,5), imshow(RGB_label), title('\fontsize{10}Connected Components');
subplot(3,4,6), imshow(filtered_labels), title('\fontsize{10}Region Property Analysis and Heuristic Filtering (Solidity > 0.5)');
subplot(3,4,7), imshow(tumor), title('\fontsize{10}Largest Region Selection');
subplot(3,4,8), imshow(tumor_dilated), title('\fontsize{10}Morphological Dilation');

subplot(3,4,9), imshow(K); hold on;
for i = 1:length(Bound)
    plot(Bound{i}(:,2), Bound{i}(:,1), 'y', 'LineWidth', 2);
end
title('\fontsize{10}Final Tumor Detection');
hold off;
