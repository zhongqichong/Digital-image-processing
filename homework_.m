function image_processing_gui
    % 创建图形界面
    f = figure('Position', [100, 100, 900, 600]);

    % 第一行按钮
    uicontrol('Style', 'pushbutton', 'String', '加载图像', ...
              'Position', [20, 450, 100, 30], ...
              'Callback', @load_image);
          
    uicontrol('Style', 'pushbutton', 'String', '直方图均衡化', ...
              'Position', [150, 450, 100, 30], ...
              'Callback', @histogram_equalization);
    
    uicontrol('Style', 'pushbutton', 'String', '灰度化', ...
              'Position', [280, 450, 100, 30], ...
              'Callback', @convert_to_grayscale);
    
    uicontrol('Style', 'pushbutton', 'String', '灰度直方图', ...
              'Position', [410, 450, 100, 30], ...
              'Callback', @show_histogram);
    
    % 对比度增强部分
    uicontrol('Style', 'text', 'String', '选择对比度增强方式:', ...
              'Position', [20, 400, 150, 30]);
          
    contrast_menu = uicontrol('Style', 'popupmenu', ...
                               'String', {'线性变换', '对数变换', '指数变换'}, ...
                               'Position', [180, 400, 120, 30]);
    
    uicontrol('Style', 'pushbutton', 'String', '增强对比度', ...
              'Position', [320, 400, 100, 30], ...
              'Callback', @(src, event) enhance_contrast(contrast_menu));

    % 新增缩放和旋转部分
    uicontrol('Style', 'text', 'String', '缩放因子:', ...
              'Position', [20, 350, 80, 30]);
    
    scale_factor_input = uicontrol('Style', 'edit', 'String', '1', ...
                                    'Position', [100, 350, 50, 30]);
    
    uicontrol('Style', 'pushbutton', 'String', '缩放图像', ...
              'Position', [160, 350, 100, 30], ...
              'Callback', @(src, event) scale_image(scale_factor_input));
    
    uicontrol('Style', 'text', 'String', '旋转角度（度）:', ...
              'Position', [300, 350, 100, 30]);
    
    rotation_angle_input = uicontrol('Style', 'edit', 'String', '0', ...
                                      'Position', [410, 350, 50, 30]);
    
    uicontrol('Style', 'pushbutton', 'String', '旋转图像', ...
              'Position', [470, 350, 100, 30], ...
              'Callback', @(src, event) rotate_image(rotation_angle_input));

    % 噪声添加部分
    uicontrol('Style', 'text', 'String', '选择噪声类型:', ...
              'Position', [20, 300, 150, 30]);
    
    noise_menu = uicontrol('Style', 'popupmenu', ...
                            'String', {'无', '高斯噪声', '椒盐噪声', '泊松噪声'}, ...
                            'Position', [180, 300, 120, 30]);
    
    uicontrol('Style', 'pushbutton', 'String', '添加噪声', ...
              'Position', [320, 300, 100, 30], ...
              'Callback', @(src, event) add_noise(noise_menu));
    
    % 滤波处理部分
    uicontrol('Style', 'pushbutton', 'String', '空间域滤波', ...
              'Position', [20, 250, 100, 30], ...
              'Callback', @spatial_filtering);
    
    uicontrol('Style', 'pushbutton', 'String', '频域滤波', ...
              'Position', [150, 250, 100, 30], ...
              'Callback', @frequency_filtering);
    
    % 第三行下拉框和按钮
    uicontrol('Style', 'text', 'String', '选择边缘检测算子:', ...
              'Position', [20, 200, 150, 30]);
    
    edge_operator_menu = uicontrol('Style', 'popupmenu', ...
                                    'String', {'Roberts', 'Prewitt', 'Sobel', 'Laplacian'}, ...
                                    'Position', [180, 200, 120, 30]);
    
    uicontrol('Style', 'pushbutton', 'String', '边缘检测', ...
              'Position', [320, 200, 100, 30], ...
              'Callback', @(src, event) edge_detection(edge_operator_menu));
    
    % 新增提取目标按钮
    uicontrol('Style', 'pushbutton', 'String', '提取目标', ...
              'Position', [20, 180, 100, 30], ...
              'Callback', @extract_target);

    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
          'Position', [120, 180, 100, 30], ...
          'Callback', @feature_extraction);

    % 显示区域
    img_axes = axes('Units', 'pixels', 'Position', [40, 50, 800, 130]);

    % 初始化图像变量
    img = [];
    noisy_img = []; % 存储噪声图像

    % 加载图像回调函数
    function load_image(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.gif', '所有图像文件'});
        if isequal(file, 0)
            return; % 用户取消
        end
        img = imread(fullfile(path, file));
        imshow(img, 'Parent', img_axes); % 显示图像
    end

    function enhance_contrast(contrast_menu)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end
        
        % 将图像转换为灰度图
        gray_img = rgb2gray(img);
        
        % 获取选中的对比度增强方式
        selected_method = contrast_menu.Value;
        
        switch selected_method
            case 1 % 线性变换
                enhanced_img = imadjust(gray_img);
                
            case 2 % 对数变换
                c = 255 / log(1 + double(max(gray_img(:))));
                enhanced_img = c * log(1 + double(gray_img));
                enhanced_img = uint8(enhanced_img);
                
            case 3 % 指数变换
                enhanced_img = 255 * (exp(double(gray_img) / 255) - 1) / (exp(1) - 1);
                enhanced_img = uint8(enhanced_img);
        end

        imshow(enhanced_img, 'Parent', img_axes); % 显示增强后的图像
    end

    function scale_image(scale_factor_input)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end
        
        scale_factor = str2double(scale_factor_input.String);
        if isnan(scale_factor) || scale_factor <= 0
            errordlg('缩放因子必须是正数！', '错误');
            return;
        end
        
        % 图像缩放
        scaled_img = imresize(img, scale_factor);
        
        % 创建新的图形窗口显示缩放后的图像
        figure('Name', '缩放后的图像', 'NumberTitle', 'off');
        imshow(scaled_img); % 显示缩放后的图像
    end

    function rotate_image(rotation_angle_input)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end
        
        rotation_angle = str2double(rotation_angle_input.String);
        if isnan(rotation_angle)
            errordlg('旋转角度无效！', '错误');
            return;
        end
        
        % 图像旋转
        rotated_img = imrotate(img, rotation_angle);
        imshow(rotated_img, 'Parent', img_axes); % 显示旋转后的图像
    end

    function add_noise(noise_menu)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end

        noise_type = noise_menu.Value;
        switch noise_type
            case 1 % 无噪声
                noisy_img = img; % 不添加噪声
            case 2 % 高斯噪声
                noisy_img = imnoise(img, 'gaussian', 0, 0.01); % 添加高斯噪声
            case 3 % 椒盐噪声
                noisy_img = imnoise(img, 'salt & pepper', 0.02); % 添加椒盐噪声
            case 4 % 泊松噪声
                noisy_img = imnoise(img, 'poisson'); % 添加泊松噪声
        end
        
        imshow(noisy_img, 'Parent', img_axes); % 显示添加噪声后的图像
    end

    function spatial_filtering(~, ~)
        if isempty(noisy_img)
            errordlg('请先添加噪声！', '错误');
            return;
        end

        % 使用均值滤波器进行空间域滤波
        filtered_img = imfilter(noisy_img, fspecial('average', [3, 3]));
        imshow(filtered_img, 'Parent', img_axes); % 显示滤波后的图像
    end

    function frequency_filtering(~, ~)
        if isempty(noisy_img)
            errordlg('请先添加噪声！', '错误');
            return;
        end

        % 将图像转换为灰度图
        if size(noisy_img, 3) == 3
            gray_img = rgb2gray(noisy_img);
        else
            gray_img = noisy_img;
        end

        % 进行傅里叶变换
        F = fft2(double(gray_img));
        Fshift = fftshift(F);

        % 创建低通滤波器
        [rows, cols] = size(gray_img);
        crow = round(rows / 2);
        ccol = round(cols / 2);
        mask = zeros(rows, cols);
        r = 30; % 设置滤波器半径
        [x, y] = meshgrid(1:cols, 1:rows);
        mask(((y - crow).^2 + (x - ccol).^2) <= r^2) = 1;

        % 应用低通滤波器
        Fshift_filtered = Fshift .* mask;

        % 进行逆傅里叶变换
        F_ishift = ifftshift(Fshift_filtered);
        filtered_img = real(ifft2(F_ishift));

        % 显示结果
        imshow(uint8(filtered_img), 'Parent', img_axes); % 显示滤波后的图像
    end

    function edge_detection(edge_operator_menu)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end
        
        % 获取选中的边缘检测算子
        selected_operator = edge_operator_menu.Value;
        
        % 根据选择的算子进行边缘检测
        switch selected_operator
            case 1 % Roberts
                edges = edge(rgb2gray(img), 'Roberts');
            case 2 % Prewitt
                edges = edge(rgb2gray(img), 'Prewitt');
            case 3 % Sobel
                edges = edge(rgb2gray(img), 'Sobel');
            case 4 % Laplacian
                edges = edge(rgb2gray(img), 'log');
        end

        imshow(edges, 'Parent', img_axes); % 显示边缘检测结果
    end

    function convert_to_grayscale(~, ~)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end

        % 检查图像是否为 RGB 图像
        if size(img, 3) == 3
            gray_img = rgb2gray(img);
        else
            gray_img = img;
            errordlg('图像已经是灰度图！', '提示');
        end

        imshow(gray_img, 'Parent', img_axes); % 显示灰度图
    end

    function show_histogram(~, ~)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end

        % 将图像转换为灰度图（如果不是的话）
        if size(img, 3) == 3
            gray_img = rgb2gray(img);
        else
            gray_img = img;
        end
        
        % 计算直方图
        [counts, bin] = imhist(gray_img);
        
        % 创建新的图形窗口显示直方图
        figure('Name', '灰度直方图', 'NumberTitle', 'off');
        bar(bin, counts, 'BarWidth', 1, 'FaceColor', 'b');
        xlim([0 255]);
        xlabel('灰度级');
        ylabel('像素数');
        title('灰度直方图');
    end

    function histogram_equalization(~, ~)
        if isempty(img)
            errordlg('请先加载图像！', '错误');
            return;
        end

        % 转换为灰度图
        gray_img = rgb2gray(img);
        % 进行直方图均衡化
        eq_img = histeq(gray_img);
        
        imshow(eq_img, 'Parent', img_axes); % 显示均衡化后的图像
    end

function extract_target(~, ~)
    if isempty(img)
        errordlg('请先加载图像！', '错误');
        return;
    end

    % 将图像转换为灰度图
    gray_img = rgb2gray(img);
    
    % 使用边缘检测算子提取边缘
    edges = edge(gray_img, 'Canny'); % 使用 Canny 边缘检测

    % 创建一个图形窗口显示提取目标的结果
    figure('Name', '提取目标', 'NumberTitle', 'off');
    imshow(edges); % 显示边缘检测结果

    % 提取轮廓
    [B, L] = bwboundaries(edges, 'noholes'); % 获取边界
    hold on;
    
    % 绘制轮廓
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2); % 绘制红色轮廓
    end
    
    hold off;
    title('提取目标');
end

    function feature_extraction(~, ~)
    figure; 
    subplot(2,2,1)
    imshow(img); title('原始图像');

    % 2. 图像灰度化（如果是彩色图像）
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    subplot(2,2,2)
    imshow(grayImg); title('灰度图像');

    % 3. 目标提取
    % 使用 Otsu 方法分割并形态学处理
    smoothImg = imgaussfilt(grayImg, 2); % 高斯滤波
    level = graythresh(smoothImg); % Otsu 阈值
    binaryImg = imbinarize(smoothImg, level); % 二值化
    cleanImg = imopen(binaryImg, strel('disk', 3)); % 开运算去噪
    filledImg = imfill(cleanImg, 'holes'); % 填充孔洞
    subplot(2,2,3)
    imshow(filledImg); title('提取目标的二值图像');

    % 提取目标区域
    labeledImage = bwlabel(filledImg); % 连通区域标记
    stats = regionprops(labeledImage, 'BoundingBox', 'Area'); % 获取属性
    largestArea = max([stats.Area]); % 找到最大区域
    largestRegion = stats([stats.Area] == largestArea); % 提取最大区域
    targetBoundingBox = largestRegion.BoundingBox; % 获取目标边界框
    targetImg = imcrop(grayImg, targetBoundingBox); % 裁剪目标区域
   subplot(2,2,4)
   imshow(targetImg, []); title('目标区域');

    % 4. LBP 特征提取
    lbpOriginal = extractLBPFeatures(grayImg); % 原始图像 LBP 特征
    lbpTarget = extractLBPFeatures(targetImg); % 目标区域 LBP 特征
    disp('原始图像 LBP 特征:');
    disp(lbpOriginal);
    disp('目标区域 LBP 特征:');
    disp(lbpTarget);

    % 可视化 LBP 特征
    figure; 
    subplot(1, 2, 1);
    plot(lbpOriginal); title('原始图像 LBP 特征');
    subplot(1, 2, 2);
    plot(lbpTarget); title('目标区域 LBP 特征');

    % 5. HOG 特征提取
    [hogOriginal, visOriginal] = extractHOGFeatures(grayImg); % 原始图像 HOG 特征
    [hogTarget, visTarget] = extractHOGFeatures(targetImg); % 目标区域 HOG 特征
    disp('原始图像 HOG 特征:');
    disp(hogOriginal);
    disp('目标区域 HOG 特征:');
    disp(hogTarget);

    % 可视化 HOG 特征
    figure; 
    subplot(1, 2, 1);
    imshow(grayImg); hold on;
    plot(visOriginal); title('原始图像 HOG 特征');
    subplot(1, 2, 2);
    imshow(targetImg); hold on;
    plot(visTarget); title('目标区域 HOG 特征');
end

end