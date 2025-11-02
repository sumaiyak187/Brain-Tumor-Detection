% Brain Tumor Detection Script (GUI-Free Version)

% === Step 1: Read MRI Image ===
imagePath = 'images\brain1.jpg';  % <-- Replace with actual path
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
        Gx = ((2*C(i+2,j+1)+ C(i+2,j)+C(i+2,j+2))-(2*C(i,j+1)+C(i,j)+C(i,j+2)));
        Gy = ((2*C(i+1,j+2)+ C(i,j+2)+C(i+2,j+2))-(2*C(i+1,j)+C(i,j)+C(i+2,j)));
        B(i,j) = sqrt(Gx.^2 + Gy.^2);
    end
end
edge_detected = B;

% === Step 4: Tumor Detection ===
bw = imbinarize(K, 0.7);
label = bwlabel(bw);
stats = regionprops(label, 'Solidity', 'Area');
density  = [stats.Solidity];
area = [stats.Area];

high_dense_area = density > 0.5;
max_area = max(area(high_dense_area));
tumor_label = find(area == max_area);
tumor = ismember(label, tumor_label);

se = strel('square', 5);
tumor = imdilate(tumor, se);

% === Draw Tumor Boundary ===
tumor_img = K;
Bound = bwboundaries(tumor, 'noholes');

% === Display Results ===
figure('Name','Brain Tumor Detection Results','NumberTitle','off');
subplot(2,2,1), imshow(original), title('\fontsize{14}Original MRI Image');
subplot(2,2,2), imshow(K), title('\fontsize{14}Median Filtered');
subplot(2,2,3), imshow(edge_detected, []), title('\fontsize{14}Edge Detection');
subplot(2,2,4), imshow(tumor_img), title('\fontsize{14}Tumor Detected'); hold on;
for i = 1:length(Bound)
    plot(Bound{i}(:,2), Bound{i}(:,1), 'y', 'LineWidth', 1.75);
end
hold off;
