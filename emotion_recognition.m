% Define the paths to train and test folders
trainFolder = 'C:\Users\ahmed\Downloads\archive\train'; % Update with your path
testFolder = 'C:\Users\ahmed\Downloads\archive\test';

% Initialize arrays to store images and labels
imageData = [];
labels = [];

% Load training data
emotionCategories = {'surprise', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'angry'};

for i = 1:length(emotionCategories)
    category = emotionCategories{i};
    imageFiles = dir(fullfile(trainFolder, category, '*.jpg')); % Assuming images are in .jpg format
    
    for j = 1:length(imageFiles)
        % Read the image
        img = imread(fullfile(trainFolder, category, imageFiles(j).name));
        
        % Check if the image is indexed or RGB
        if size(img, 3) == 3  % RGB image
            imgGray = rgb2gray(img); % Convert to grayscale
        elseif size(img, 3) == 1  % Grayscale image
            imgGray = img; % Already grayscale
        else
            % If the image has more than 3 channels, use im2gray
            imgGray = im2gray(img);
        end
        
        % Preprocess: Resize
        imgGray = imresize(imgGray, [48 48]); % Resize to 48x48
        
        % Normalize
        imgNormalized = double(imgGray) / 255;
        
        % Store the image and label
        imageData(:, end+1) = imgNormalized(:); % Append normalized image (flattened)
        labels(end+1) = i; % Append label (i corresponds to the emotion category)
    end
end

% Reshape imageData for training
% No need to reshape, already in the correct format

% Train a classifier
classifier = fitcecoc(imageData', categorical(labels)); % Using ECOC for multi-class classification

% Save the trained classifier
save('emotionClassifier.mat', 'classifier');

disp('Training complete and classifier saved.');

% Real-time Emotion Recognition Using Webcam

% Load the trained classifier
load('emotionClassifier.mat');

% Initialize the webcam
cam = webcam;

% Create a face detector
faceDetector = vision.CascadeObjectDetector();

disp('Starting real-time emotion recognition. Press any key to exit.');

while true
    % Capture frame
    frame = snapshot(cam);
    
    % Detect faces
    bbox = step(faceDetector, frame);
    
    % Process each detected face
    for i = 1:size(bbox, 1)
        % Extract the face region
        face = imcrop(frame, bbox(i, :));
        
        % Check if the face is indexed or RGB
        if size(face, 3) == 3  % RGB image
            faceGray = rgb2gray(face); % Convert to grayscale
        elseif size(face, 3) == 1  % Grayscale image
            faceGray = face; % Already grayscale
        else
            faceGray = im2gray(face); % Handle more than 3 channels
        end
        
        % Preprocess the face
        faceGray = imresize(faceGray, [48 48]); % Resize to 48x48
        faceNormalized = double(faceGray) / 255; % Normalize
        
        % Predict emotion
        predictedLabel = predict(classifier, faceNormalized(:)');
        
        % Annotate the image with the predicted emotion
        frame = insertText(frame, bbox(i, 1:2), char(emotionCategories{predictedLabel}), ...
                           'TextColor', 'white', 'FontSize', 18);
    end
    
    % Display the annotated frame
    imshow(frame);
    
    % Exit condition (e.g., press a key)
    if ~isempty(get(gcf, 'CurrentKey'))
        break;
    end
end

% Clean up
clear cam;
disp('Exiting...');