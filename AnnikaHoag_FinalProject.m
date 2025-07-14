% Annika Hoag
% COM 322 Computer Version - Final Project
% Program that runs OpenPose on images of ballet poses done by 3 dancers,
% reads in joint positions from OpenPose and sets up a matrix with the
% relevant data to be used in training via MatLab's Classification Learner
% App (specifically with Quadratic SVM)


% Install OpenPose https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_installation_0_index.html 
% Run OpenPose as a terminal command
% Notes: 
%   1) this file must be in the openpose directory for this commmand to
% work, 
%   2) the display must be on for the keypoint files to be written
%   correctly
command= '.\bin\OpenPoseDemo.exe --image_dir .\trainData --write_json .\keypoints --model_pose BODY_25';
[status, cmdout] = system(command);


% Call the function to set up the data for Classification Learner training
% using my handcrafted features (uncommented Line 23) or with all features 
% given as data (commented out Line 24)   
features = classifyPosesHandcrafted();
%features = classifyPoses();
CLASSIFICATION_LEARNER_MODEL = true;


% Other functions (see below)

% To run plotSkeleton, comment out Lines 16 and 17 and then uncomment the
% following 3 lines 
% command = '.\bin\OpenPoseDemo.exe --image_dir .\images --write_json .\keypoints --model_pose BODY_25 ';
% [status, cmdout] = system(command);
% plotSkeleton();

% To run plotFeatures, comment out Lines 16 and 17 and then uncomment the
% following 3 lines
% command = '.\bin\OpenPoseDemo.exe --image_dir .\groundTruths --write_json .\keypoints --model_pose BODY_25 ';
% [status, cmdout] = system(command);
% plotFeatures();



% Function to do data set up for training with MatLab's Classification 
% Learner using my handcrafted features
% Has about mid 90s for accuracy (95%-97%)
function  f=classifyPosesHandcrafted()

    % Read in images 
    fn = dir('trainData\*.jpg'); 
    imgs = size(fn,1);
 
    % Feature matrix, each row is an image, each column is a feature, 
    % last column is the class the pose in the image belongs to
    % Features are distance from mid hip to wrists and ankles, confidence,  
    % distance between wrists, and distance from each ankle to each wrist
    NUM_FEATURES = 13;
    features = zeros(imgs, NUM_FEATURES+1);


    % Loop through images, calculate features, and put those features into 
    % the matrix and the string of the image name into the last column
    for image_idx = 1 : imgs

        % Get keypoint file name and read in data from it 
        keypointName = extractBetween(fn(image_idx).name, 1, size(fn(image_idx).name,2)-4) + "_keypoints.json";
        str = fileread('keypoints\'+keypointName);
        data = jsondecode(str).people.pose_keypoints_2d;

         % Hard code key point values for necessary joints since the Body
         % 25 model we're using has a standard order of the joints
        hipX = data(25);
        hipY= data(26);
        rWristX = data(13);
        rWristY = data(14);
        rWristConfidence = data(15);
        lWristX = data(22);
        lWristY = data(23);
        lWristConfidence = data(24);
        rAnkleX = data(34);
        rAnkleY = data(35);
        rAnkleConfidence = data(36);
        lAnkleX = data(43);
        lAnkleY = data(44);
        lAnkleConfidence = data(45);

        % Find distances between mid hip and both wrists and both ankles,
        % and distances from both ankles to both wrists using my distance
        % function 
        hipToRWrist = euclideanDistance(hipX, hipY, rWristX, rWristY);
        hipToLWrist = euclideanDistance(hipX, hipY, lWristX, lWristY);
        hipToRAnkle = euclideanDistance(hipX, hipY, rAnkleX, rAnkleY);
        hipToLAnkle = euclideanDistance(hipX, hipY, lAnkleX, lAnkleY);
        wristToWrist = euclideanDistance(rWristX, rWristY, lWristX, lWristY);
        rAnkleToRWrist = euclideanDistance(rAnkleX, rAnkleY, rWristX, rWristY);
        lAnkleToLWrist = euclideanDistance(lAnkleX, lAnkleY, lWristX, lWristY);
        rAnkleToLWrist = euclideanDistance(rAnkleX, rAnkleY, lWristX, lWristY);
        lAnkleToRWrist = euclideanDistance(lAnkleX, lAnkleY, rWristX, rWristY);

        % Add to feature matrix
        features(image_idx, :) = [hipToRWrist, rWristConfidence, hipToLWrist, lWristConfidence, hipToRAnkle, rAnkleConfidence, hipToLAnkle, lAnkleConfidence, wristToWrist, rAnkleToRWrist, lAnkleToLWrist, rAnkleToLWrist, lAnkleToRWrist, str2double(fn(image_idx).name(1))];
  
    end 

    % Return feature matrix
    f = features;
end




% Function to do data set up for training with MatLab's Classification 
% Learner with features being every joint in relation to hip to see how the
% classifier does without specifically chosen features
% Has about low 90s or high80s for accuracy (around 90%)
function f=classifyPoses()

    % Read in images
    fn = dir('trainData\*.jpg'); 
    imgs = size(fn,1);
 
    % Feature matrix, each row is an image, each column is a feature, 
    % last column is the class the pose in the image belongs to
    % Features are distance from mid hip to every other joint
    NUM_FEATURES = 27;
    features = zeros(imgs, NUM_FEATURES+1);

  
    % Loop through images, calculate features, and put those features into 
    % the matrix and the string of the image name into the last column
    for image_idx = 1 : imgs

        % Get keypoint file name and read in data from it
        keypointName = extractBetween(fn(image_idx).name, 1, size(fn(image_idx).name,2)-4) + "_keypoints.json";
        str = fileread('keypoints\'+keypointName);
        data = jsondecode(str).people.pose_keypoints_2d;

        % Hard code key point values for all joints since the Body
        % 25 model we're using has a standard order of the joints
        hipX = data(25);
        hipY= data(26); 
        noseX = data(1);
        noseY = data(2);
        neckX = data(4);
        neckY = data(5);
        rShoulderX = data(7);
        rShoulderY = data(8);
        rElbowX = data(10);
        rElbowY = data(11);
        rWristX = data(13);
        rWristY = data(14);
        lShoulderX = data(16);
        lShoulderY = data(17);
        lElbowX = data(19);
        lElbowY = data(20);
        lWristX = data(22);
        lWristY = data(23);
        rHipX = data(28);
        rHipY = data(29);
        rKneeX = data(31);
        rKneeY = data(32);
        rAnkleX = data(34);
        rAnkleY = data(35);
        lHipX = data(37);
        lHipY = data(38);
        lKneeX = data(40);
        lKneeY = data(41);
        lAnkleX = data(43);
        lAnkleY = data(44);
        rEyeX = data(46);
        rEyeY = data(47);
        lEyeX = data(49);
        lEyeY = data(50);
        rEarX = data(52);
        rEarY = data(53);
        lEarY = data(55);
        lEarX = data(56);
        lBigToeX = data(58);
        lBigToeY = data(59);
        lSmallToeX = data(61);
        lSmallToeY = data(62);
        lHeelX = data(64);
        lHeelY = data(65);
        rBigToeX = data(67);
        rBigToeY = data(68);
        rSmallToeX = data(70);
        rSmallToeY = data(71);
        rHeelX = data(73);
        rHeelY = data(74);
 
        % Find distances between mid hip and all joints as hardcoded above
        hipToNose = euclideanDistance(hipX, hipY, noseX, noseY);
        hipToNeck = euclideanDistance(hipX, hipY, neckX, neckY);
        hipToRShoulder = euclideanDistance(hipX, hipY, rShoulderX, rShoulderY);
        hipToRElbow = euclideanDistance(hipX, hipY, rElbowX, rElbowY);
        hipToRWrist = euclideanDistance(hipX, hipY, rWristX, rWristY);
        hipToLShoulder = euclideanDistance(hipX, hipY, lShoulderX, lShoulderY);
        hipToLElbow = euclideanDistance(hipX, hipY, lElbowX, lElbowY);
        hipToLWrist = euclideanDistance(hipX, hipY, lWristX, lWristY);
        hipToRHip = euclideanDistance(hipX, hipY, rHipX, rHipY);
        hipToRKnee = euclideanDistance(hipX, hipY, rKneeX, rKneeY);
        hipToRAnkle = euclideanDistance(hipX, hipY, rAnkleX, rAnkleY);
        hipToLHip = euclideanDistance(hipX, hipY, lHipX, lHipY);
        hipToLKnee = euclideanDistance(hipX, hipY, lKneeX, lKneeY);
        hipToLAnkle = euclideanDistance(hipX, hipY, lAnkleX, lAnkleY);
        hipToREye = euclideanDistance(hipX, hipY, rEyeX, rEyeY);
        hipToLEye = euclideanDistance(hipX, hipY, lEyeX, lEyeY);
        hipToREar = euclideanDistance(hipX, hipY, rEarX, rEarY);
        hipToLEar = euclideanDistance(hipX, hipY, lEarX, lEarY);
        hipToLBigToe = euclideanDistance(hipX, hipY, lBigToeX, lBigToeY);
        hipToLSmallToe = euclideanDistance(hipX, hipY, lSmallToeX, lSmallToeY);
        hipToLHeel = euclideanDistance(hipX, hipY, lHeelX, lHeelY);
        hipToRBigToe = euclideanDistance(hipX, hipY, rBigToeX, rBigToeY);
        hipToRSmallToe = euclideanDistance(hipX, hipY, rSmallToeX, rSmallToeY);
        hipToRHeel = euclideanDistance(hipX, hipY, rHeelX, rHeelY);

        % Add to feature matrix
        features(image_idx, :) = [hipToNose, hipToNeck, hipToRShoulder, hipToRElbow, hipToRWrist, hipToLShoulder, hipToLElbow, hipToLWrist, hipToRHip, hipToRKnee, hipToRAnkle, hipToLHip, hipToLKnee, hipToLAnkle, hipToREye, hipToLEye, hipToREar, hipToLEar, hipToLBigToe, hipToLSmallToe, hipToLHeel, hipToRBigToe, hipToLSmallToe, hipToLHeel, hipToRBigToe, hipToRSmallToe, hipToRHeel, str2double(fn(image_idx).name(1))];
       
    end 

    % Return feature matrix
    f = features;
end



% Euclidean distance helper function to be used in getting features
function dist=euclideanDistance(x1, y1, x2, y2)
    dist = sqrt((x2-x1)^2 + (y2-y1)^2);
end



%-------Below are some functions I wrote in the stages of figuring out how
%-------to read the keypoint files and what features to use if you want to
% run those, they are commented out above with instructions on how to run


% Plotting the keypoints written to json files, on the corresponding 
% images, from running OpenPose to check if the keypoints were written correctly 
% Note that this function was written before I cleaned up the data for
% training so it is running this with the images directory
function plotSkeleton()
    
    % Read in images
    fn = dir('images\*.jpg'); % read all images in the folder
    total_images = size(fn,1);
    
    % Loop through every 25 images (to save time) and plot the keypoints on them 
    for image_idx = 1 : 25 : total_images
        % Get image
        image = imread(strcat(fn(image_idx).folder, '\', fn(image_idx).name) );
        image = imrotate(image, -90);
        imshow(image);
    
        % Get keypoint file name and read in data from it 
        keypointName = extractBetween(fn(image_idx).name, 1, 8) + "_keypoints.json";
        str = fileread("keypoints\"+keypointName);
        data = jsondecode(str).people.pose_keypoints_2d;
        len = size(data, 1);
    
        % Loop through json data and plot keypoints on the image, also scale
        % marker size based on the confidence of that keypoint 
        imshow(image);
        hold on;
        for i = 3 : 3 : len
            % Only plot if the confidence is above 0, i.e. the joint was found 
            if data(i) ~= 0
                plot(data(i-2), data(i-1), "r.", "MarkerSize", data(uint16(i))*35);
            end
            pause(1);
        end 
        hold off;
    end

end



% Graphing potential features to determine if the ones I'm trying are
% distinguishable enough
% Note that this function was written to be run with the groundTruths
% folder containing only 1 image for each of the original 8 classes
% The results from this function is what made me decide to combine 2
% classes into 1
function plotFeatures()

    % Read in images
    fn = dir('groundTruths\*.jpg'); 
    ground_truths = size(fn,1);

    % Hard code necessary keypoint value matrices
    hipRWrist = zeros(size(ground_truths, 1));
    hipLWrist = zeros(size(ground_truths, 1));
    hipRAnkle = zeros(size(ground_truths, 1));
    hipLAnkle = zeros(size(ground_truths, 1));
    ankleHeights = zeros(size(ground_truths, 1));

    % Loop through the images
    for image_idx = 1 : ground_truths

        % Get keypoint file name and read in data from it 
        keypointName = extractBetween(fn(image_idx).name, 1, size(fn(image_idx).name,2)-4) + "_keypoints.json";
        str = fileread('keypoints\'+keypointName);
        data = jsondecode(str).people.pose_keypoints_2d;
    
        % Hard code necessary key point values 
        hipX = data(24);
        hipY= data(25);
        rWristX = data(12);
        rWristY = data(13);
        lWristX = data(21);
        lWristY = data(22);
        rAnkleX = data(33);
        rAnkleY = data(34);
        lAnkleX = data(42);
        lAnkleY = data(43);

        % 1st idea --> Find distances between mid hip and both wrists and both ankles
        hipToRWrist = euclideanDistance(hipX, hipY, rWristX, rWristY);
        hipToLWrist = euclideanDistance(hipX, hipY, lWristX, lWristY);
        hipToRAnkle = euclideanDistance(hipX, hipY, rAnkleX, rAnkleY);
        hipToLAnkle = euclideanDistance(hipX, hipY, lAnkleX, lAnkleY);

        % Add each distance to its respective list
        hipRWrist(image_idx) = hipToRWrist;
        hipLWrist(image_idx) = hipToLWrist;
        hipRAnkle(image_idx) = hipToRAnkle;
        hipLAnkle(image_idx) = hipToLAnkle;

        % 2nd idea --> Seeing if one ankle being higher than the other is a
        % feature
        lAnkleRAnkleHeight = euclideanDistance(lAnkleX, lAnkleY, rAnkleX, rAnkleY);
        ankleHeights(image_idx) = lAnkleRAnkleHeight;
    end

    % Plot the distances as lines in a graph
    plot(hipRWrist, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hip to RWrist'); 
    hold on 
    plot(hipLWrist, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hip To LWrist'); 
    plot(hipRAnkle, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hip To RAnkle'); 
    plot(hipLAnkle, '-*', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Hip To LAnkle');
    plot(hipLAnkle, '-+', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Ankle Heights');
    xlabel('Index/Pose');
    xticklabels({'ALaQuatriemeDerriere', 'ALaQuatriemeDevant', 'ALASeconde', 'CroiseDerriere', 'CroiseDevant', 'Ecarte', 'Efface', 'Epaule'});
    ylabel('Measurement');

    legend show; % Display the legend for each line

    hold off;
end
