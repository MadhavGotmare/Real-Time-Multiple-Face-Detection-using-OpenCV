REAL-TIME MULTIPLE FACE RECOGNITION USING OPENCV

VIth SEMESTER  
COMPUTER ENGINEERING 

Submitted By
Bhakti Nagrecha
Madhav Gotmare
Prince Pille
Salman Rizvi

Under the guidance of 
Prof. Vaibhav Deshpande

Academic Year 2019 – 2020
Department of Computer Engineering
 
ST. VINCENT PALLOTTI COLLEGE OF ENGINEERING AND TECHNOLOGY 
Wardha Road, Gavsi Manapur, Nagpur 
ST. VINCENT PALLOTTI COLLEGE OF ENGINEERING AND TECHNOLOGY
Wardha Road, Gavsi Manapur, Nagpur

Department of Computer Engineering










CONTENT

Abstract………………………………………………………….	I
1.	INTRODUCTION…………………………………………...	1
	1.1	OVERVIEW……………………………………………	1
	1.2	FACE RECOGNITION………………………………..	2
	1.3	FACE DETECTION…………………………...............	4
2.	LITERATURE SURVEY…………………………………...	6
	2.1	OVERVIEW……………………………………………	6
	2.2	OVERVIEW OF DETECTION CLASSIFICATION…………………………………….	
6
	2.3	BASIC CONCEPTS USED BY MOST FACE DETECTION METHOD………………………………	
8
	2.4	METHOD OF FACE DETECTION…………………...	10
3.	PROJECT PLANNING AND SCHEDULING……………..	13
4.	REQUIREMENT ANAYLSIS……………………………...	14
	4.1	FUCTIONAL REQUIREMENT……………………….	14
	4.2	NON-FUNCTIONAL REQUIREMENT………………	15
5.	SYSTEM DESIGN AND IMPLEMENTATION…………...	16
	5.1	OVERVIEW……………………………………………	16
	5.2	IMPORT THE REQUIRED MODULES……………...	16
	5.3	LOAD THE FACE DETECTION CASCADE………..	17
	5.4	CREATE THE FACE RECOGNIZER OBJECT……...	19
	5.5	PERFORM TRAINING………………………………..	23
	5.6	TESTING THE MODULE…………………………….	23
6.	TESTING……………………………………………………	24
	6.1	OVERVIEW……………………………………………	24
	6.2	TESTING ON DIFFERENT NUMBER OF FACES….	24
7.	CONCLUSION AND FUTURE WORK…………………...	26
REFERENCE……………………………………………………	27
















ABSTRACT


In the past years, a lot of effort has been made in the field of face detection. The human face contains important features that can be used by vision-based automated systems in order to identify and recognize individuals. The face is one of the easiest ways to distinguish the individual identity of each other. Face recognition is a personal identification system that uses personal characteristics of a person to identify the person's identity.

In this project we intend to implement a Real Time Face Recognition, that can be performed in two stages such as, Face detection and Face recognition. In this project we implemented “Haar-Cascade algorithm” to identify human faces which is organized in Open CV by Python language and “Local Binary Pattern Histogram algorithm” to recognize faces. Collating with other existing algorithms, this classifier produces a high recognition rate even with varying expressions, efficient feature selection and low assortment of false positive features. Haar feature-based cascade classifier system utilizes only 200 features out of 6000 features to yield a recognition rate of 85-95%.  

Keywords: - OpenCV, Face Recognition, Haar Cascade, LBPH. 
 
CHAPTER NO. 1
INTRODUCTION
1.1	OVERVIEW
Face recognition is a biometric software application adapted to identify individuals via tracking and detecting. The main intention of this project is to recognize the faces of people. This approach can be executed practically in crowded areas like airports, railway stations, universities and malls. 
Although recognizing an individual by the face is an easy task for humans, it is a challenge for vision-based automated systems. It has been an active research area involving several disciplines such as image processing, neural networks, statistics, pattern recognition, anthropometry and computer vision. Vision-based automated systems can apply facial recognition and facial identification in numerous commercial applications, such as biometric authentication, human-computer interaction, surveillance, games and multimedia entertainment. 
Unlike other biometrics, face recognition is non-invasive, and does not need physical contact of the individual with the system, making it a very acceptable biometric. 
This project focuses on automatic face recognition and extracting those meaningful features from an image, putting them into a useful representation and performing some classifications on them. Face recognition based on the geometric features of a face is probably the most intuitive approach to Human identification. The whole process can be divided in three major steps where the first step is to find a good database of faces with multiple images for each individual. The next step is to detect faces in the database images and use them to train the face recognizer and the last step is to test the face recognizer to recognize faces it was trained for.

1.2	FACE RECOGNITION
DIFFERENT APPROCHES OF FACE RECOGNITION
There are two predominant approaches to the face recognition problem: Geometric (feature based) and Photometric (view based). As research interest in face recognition continued, many different algorithm algorithms were developed, 3 have been well studied for face recognition literature.
Recognition algorithm can be divided into two main approaches:
1.2.1	Geometric:  Is based on geometrical relationship between facial landmarks, or in other words the spatial configuration of facial features. That means that the main geometrical features of the face such as the eyes, nose and mouth are first located, and the faces are classified on the basis of various geometrical distances and angles between features. (see figure 1.1)
Figure 1.1: Geometric Facial Recognition

1.2.2	Photometric: Used to recover the shape of an object from several images taken under different lighting conditions. The shape of the recovered object is defined by a gradient map, which is made up of an array of surface normal. (see figure 1.2)
Figure 1.2: Photometric Stereo Image
Popular recognition algorithm includes: -
a)	Principal Component Analysis using Eigenfaces (PCA)
b)	Haar Cascade Algorithm
c)	Elastic Bunch Graph Matching using the Fisherface Algorithm.

1.3	FACE DETECTION
Face detection involves separating image windows into two classes: one containing faces (taming the background (clutter). It is difficult because although commonalities exist between faces, they can vary considerably in terms of age, skin color and facial expression. The problem is further complicated by differing lighting conditions, image qualities and geometries, as well as the possibility of partial occlusion and disguise. An ideal face detector would therefore be able to detect the presence of any face under any set of lighting conditions, upon any background.
The face detection task can be broken down into two steps: -
STEP I: Is a classification task that takes some arbitrary image as input and outputs a binary value of yes or no, indicating whether there are any faces present in the image.
STEP II: Is the face localization task that aims to take an image as input and output the location of any face or faces within that image as some bounding box with (x, y, width, height).

The face detection system can be divided into the following steps: 
1.3.1	Pre-Processing: - To reduce the variability in the faces, the images are processed before they are feed into the network. All positive examples that is the face images are obtained by cropping images with frontal faces to include only the front view. All the cropped images are then corrected for lighting through standard algorithm.
1.3.2	Classification: - Neural networks are implemented to classify the images or faces or non faces by training on these examples. We use implementation of the neural network for this task. Different network configuration is experimented to optimize the result.
1.3.3	Localization: - The trained neural network is then used to search for faces in an image and if present localize them in a bounding box. Various features of face on which network the work has been done on position scale Orientation illumination.






Figure 1.3: Face Detection Algorithm
CHAPTER NO. 2
LITERATURE SURVEY
2.1	OVERVIEW
Face Detection is regular and almost effortless task for human beings. But for computer/machine to identify the faces in each scenario the task is not that simple. The main aim of any face capturing device is that of face recognition. The very first step of Face Recognition is Face Detection. In this section we would be discussing about various methodologies employed in Face Detection. 
The main aim of face detection can be broken down in two steps: -
A.	To find out whether there is any face in a given image or not.
B.	If, yes then where is it located. There are several factors that makes face detection complicated in an image. They are profile pose, titled pose, double chin, facial expression, hair-do, occlusion, low-resolution, out of focus faces etc. which require different computation while detecting.

2.2	OVERVIEW OF DETECTION CLASSIFICATION

Most of the Classification uses Feature Based Approach but they then differ in the way they use other different techniques and make related decisions on it. Taking this fact into account we classify different approaches.
2.2.1	Feature Based Approach: This approach relies on extraction of facial features to detect face.
A.	Low Level Analysis: It uses the concept of pixel analysis, edge detection in image (using Sobel or Canny) and gray scale. It also uses the concept of finding local maxima (to detect the nose) and local minima (to detect eyebrows pupils and lips).
B.	Feature Analysis: It improves the result of Low-Level Analysis. It incorporates the fact that all the parts of faces (eyes, nose, mouth, chin, head-top) are somewhat at relative positions with respect to each other. Prominent features (mentioned above) are determined and they in result help in identifying potential face.

2.2.2	Geometry Based Detection: It too uses the concept of Edge Detection using the concept of Canny filter and gradient analysis. All the predominate features/specific location of image is divided into block and each block has a corresponding pixel at center. All the central pixels of blocks are connected to nearby central pixels with an aim to span the face.

2.2.3	Appearance Based Approach: This approach relies on extraction of facial features to detect face. In this method entire image is processed in 2-Dimensions. All the extracted characteristics are termed as features. In order to identify the face from the given image we would be required to match only those above features that correspond to the features of human face (nose, eyes, mouth etc.). To extract the feature vector, Principle Component Analysis (PCA) and Independent Component Analysis (ICA) is used. We are using PCA because as the name suggest it would only compute or retain important/predominate vectors/variables and would reject the ones that do not contribute to any new information. This results in reducing computing and Time Complexity.

2.3	BASIC CONCEPTS USED BY MOST FACE DETECTION METHOD

2.3.1	Haar Like Features: Initially to detect a face we were directly computing pixels. This features though exhaustive is also computationally not viable as in an HD image it would result in 1920 x 1080 = 2 x 106 pixels. Thus, we moved on the feature extraction from pixel computation.
Entire human race possess face that has similar properties. The properties we refer to here are the positioning of eyes, nose, mouth etc. The relative size of them and the contrast/intensity of them. This uniformity of features can be replicated using features known as Haar-like features.
A Haar-like feature consists of adjacent rectangular windows at specific location. It adds the pixel intensities in each region and then calculates the difference of both regions. The output value is categorizing this specific location. For example, Region of Eyes is darker than cheeks. Thus, the Haar feature for it would incorporate two adjacent rectangles. One on eyes and another below it, on cheeks. Then the summation of intensities is done for each rectangular and then value of summation of rectangle on cheeks is subtracted from the sum of values in rectangle on eyes. The same concept is used for identification of eyes, mouth and bridge of nose.

2.3.2	Adaboost: Features computation using the concept of Haar like features helps identify specific region. But there are vast numbers of features for example there are about 1,80,000 features in 24 x 24-pixel window. This would undoubtedly result in large scale computation and ultimately in high time complexity.
But of these lakhs of features there are only selected features that would help predict face with better accuracy. In general terms, there are only selected features that are necessary to build a model/algorithm that detects the face with required accuracy.
Adaboost is used for this very purpose. It selects the few necessary features which when combined together/amalgamated provides a classifier that is effective for the classification of face/required object in an image. What makes Adaboost applicable in different scenarios is the fact that it is adaptive in nature. Subsequent classifiers are built to modify and improve on those cases that were misclassified by previous classifier.


2.4	METHODS OF FACE DETECTION
2.4.1	Annotation of face using ellipse: Annotating face in shape of ellipse. Technique used: Three features are required to be extracted for face detection. They are, head-top, chin and pair-of-eyes. The distance between chin and head-top is taken as the length of major axis. The length of two eyes from one’s end point to another and then some other value added to it is taken as the length of minor axis. From the length of major and minor axis we create ellipse which is approximated to encompass human face (it does not include ears as part of face). This technique requires modification when dealing with irregular face poses. That is, the faces that contain double chin, hairdo, facial expression and occlusion. Though the basic concept of minor and major axis remains the same, the way of calculating it differs.
Faces that are not to be considered: Faces looking away from the camera are considered as non-face region. Face with non-visible two eyes is not to be considered. Also, the faces are rejected where position, size and orientation are not clearly visible.

2.4.2	Viola-Jones: Viola-Jones uses the concept of Haar-Like Features, Cascade Filtering and Adaboost. For face detection this algorithm goes through three stages:
A.	Computational of Integral Images: It uses the concept of Haar-Like features. Rather, it computes the Haar-Like features through the concept of Integral Image. Integral image is a name given to the concept of Summed Area Table (both a data structure and an Algorithm) which is used to effectively and efficiently compute the sum of values in a rectangular subset of grid. Thus, the concept of Integral Image computes rectangular features (Haar Features) in constant time. The integral image at location (x, y) is the sum of the pixels above and to the left of (x, y), inclusive.

B.	Usage of Adaboost Algorithm: From vast number of features computed, we are interested in only selected few features that would enable us to detect face with great accuracy. For this, we use Adaboost Algorithm to select principal features and to train classifiers that would be using them. Aim of this algorithm is to create strong classifier from linear combination of weak classifier.


C.	Creating Cascade Structure: This cascade structure consists of classifiers. It works in a manner that initial classifiers are simpler, and they are used to reject majority of sub-windows and at end complex classifiers are used to achieve low false positive rates. The classifiers are trained using the above concept of Adaboost Algorithms. The deeper we go in the cascade the more difficult the task of the classifier is.
On referring various papers, we come to understand the challenges faced in Face Detection and the various methodologies used to detect face. From this Literature Survey we have following take-aways:
1.	It is very important to remove background information. Removing irrelevant information, such as noise and non-face part would make face detection less complicated.
2.	Feature based analysis is one of the predominant methodologies that most of the Detection Algorithms use in one way or another. Hence, efficient feature selection is very crucial.
3.	We must choose at-least two features for face identification. Because, depending only on one feature might result in erroneous detection.
4.	Varied Facial Expression and poses makes face detection more complicated.
5.	Lightning conditions greatly affects face detection.
6.	Computations need to be fast and should require less main memory as majority of application are of real time in nature.
7.	When going through the cascade like methodology, re-computation of an already computed face must be avoided.
8.	It is very essential for a methodology to define its definition of face and successful face detection.







CHAPTER NO. 3
PROJECT PLANNING AND SCHEDULING

Figure 3.1: Planning & Scheduling for month of March

Figure 3.2: Planning & Scheduling for month of April



CHAPTER NO. 4
REQUIREMENT ANALYSIS
4.1	FUNCTIONAL REQUIREMENT
Functional requirement will specify a behavior or function of the system.
A.	Design and Implementation constraints: -
•	Easy access and higher accuracy of face recognition.
•	System should recognize maximum number of faces.

B.	Hardware requirements (Minimum): - 
•	Requires a 64-bit processor and operating system.
•	Windows 7/8.1/10 (64-bit versions).
•	Intel Core i5-6600k 3.5 GHz/ AMD Ryzen 3 1300X 3.5 GHz or equivalent.
•	8 GB RAM.
•	NVIDIA GTX  780 3GB/ AMD Radeon R9 285 2GB.
•	External Camera 15 Megapixel.

C.	Software requirements: - 
•	PyCharm (IDE)
•	OpenCV 3.4.3
•	Haar Cascade Directories
•	NumPy
4.2	NON – FUNCTIONAL REQUIREMENT
Non – Functional requirement specify how the system should behave and that it is constraints upon the system behavior. The Non – Functional requirements of our system are.

A.	Performance: - The system should perform effectively and should produce less overhead.

B.	Reliability: - There must not be any duplication or interference in the data. The data reliability must be checked.

C.	Scalability: - The system should work with the same efficiency in all the devices; independent of the device version of operating system.







CHAPTER NO. 5
SYSTEM DESIGN AND IMPLEMENTATION
5.1	OVERVIEW
On External Camera we stream a live stream video. The captured image from the camera will get detected first and then cropped in Region of Interest (ROI) to reach the computer. Using a python-based Open CV library this detection is performed through Viola-Jones Haar cascade classifier, then face is trained and saved in Database File. Whenever an image arrives, the script commences LBPH algorithm on this face, evens-up the image to minimize the variations and finally compares the emanated LBPH from detected image with the pre-saved LBPH in the database. Thus, the Face is Recognized.

5.2	IMPORT THE REQUIRED MODULES
The modules required to perform the facial recognition are cv2, OS, image module and NumPy. cv2 is the OpenCV module and contains the functions for face detection and recognition. OS will be used to maneuver with image and directory names. First, we use this module to extract the image names in the database directory and then from these names individual number is extracted, which is used as a label for the face in that image. Since, the dataset images are in gif format and as of now, OpenCV does not support gif format, Image module from PIL is used to read the image in grayscale format. NumPy arrays are used to store the images.


Figure 5.1: How OpenCV face recognition works

5.3	LOAD THE FACE DETECTION CASCADE
To Load the face detection, Cascade the first step is to detect the face in each image. Once we get the region of interest containing the face in the image, we use it for training the recognizer. For the purpose of face detection, we will use the Haar Cascade provided by OpenCV. The Haar Cascades that come with OpenCV are located in the directory of OpenCV installation. Haar Cascade frontal face default.xml is used for detecting the face. Cascade is loaded using the cv2 Cascade Classifier function which takes the path to the cascade xml file. If the xml file is in the current working directory, then relative path is used.

Figure 5.2: Haar Cascade features

Haar Cascade is based on the Haar Wavelet technique to analyze pixels in the image into squares by function. This uses machine learning techniques to get a high degree of accuracy from what is called “training data”. This uses “integral image” concepts to compute the “features” detected. Haar Cascades use the Adaboost learning algorithm which selects a small number of important features from a large set to give an efficient result of classifiers.

Fig 5.3: Feature Extraction from Haar Cascade Classifier
Haar Cascades use machine learning techniques in which a function is trained from a lot of positive and negative images. This process in the algorithm is feature extraction.

5.4	CREATE THE FACE RECOGNIZER OBJECT
The next step involves creating the face recognizer object. The face recognizer object has functions like faceRecognizer_train to train the recognizer and faceRecognizer_predict to recognize a face. OpenCV currently provides Eigenface Recognizer, Fisherface Recognizer and Local Binary Patterns Histograms Face Recognizer. We have used Local Binary Patterns Histograms Face Recognizer to perform face recognition. With Local Binary Patterns it is possible to describe the texture and shape of a digital image. This is done by dividing an image into several small regions from which the features are extracted that can be used to get a measure for the similarity between the images.
Applying the LBP operation: The first computational step of the LBPH is to create an intermediate image that describes the original image in a better way, by highlighting the facial characteristics. To do so, the algorithm uses a concept of a sliding window, based on the parameter’s radius and neighbors.
The image below shows this procedure:
Figure 5.4: LBPH operation on a face
Based on the image above, let’s break it into several small steps so we can understand it easily:
•	Suppose we have a facial image in grayscale.
•	We can get part of this image as a window of 3x3 pixels.
•	It can also be represented as a 3x3 matrix containing the intensity of each pixel (0~255).
•	Then, we need to take the central value of the matrix to be used as the threshold.
•	This value will be used to define the new values from the 8 neighbors.
•	For each neighbor of the central value (threshold), we set a new binary value. We set 1 for values equal or higher than the threshold and 0 for values lower than the threshold.
•	Now, the matrix will contain only binary values (ignoring the central value). We need to concatenate each binary value from each position from the matrix line by line into a new binary value (e.g. 10001101). Note: some authors use other approaches to concatenate the binary values (e.g. clockwise direction), but the final result will be the same.
•	Then, we convert this binary value to a decimal value and set it to the central value of the matrix, which is actually a pixel from the original image.
•	At the end of this procedure (LBP procedure), we have a new image which represents better the characteristics of the original image.

NOTE: The LBP procedure was expanded to use a different number of radius and neighbors, it is called Circular LBP.
 
Figure 5.5: Features of LBPH
It can be done by using bilinear interpolation. If some data point is between the pixels, it uses the values from the 4 nearest pixels (2x2) to estimate the value of the new data point.
Extracting the Histograms using the image generated in the last step, we can use the Grid X and Grid Y parameters to divide the image into multiple grids, as can be seen in the following image:
 
Figure 5.6: Histogram on each region
Based on the image above, we can extract the histogram of each region as follows:
•	As we have an image in grayscale, each histogram (from each grid) will contain only 256 positions 0~255) representing the occurrences of each pixel intensity.
•	Then, we need to concatenate each histogram to create a new and bigger histogram. Supposing we have 8x8 grids, we will have 8x8x256=16.384 positions in the final histogram. 

5.5	PERFORM THE TRAINING
To create the function to prepare the training set, we will define a function that takes the absolute path to the image database as input argument and returns tuple of 2 list, one containing the detected faces and the other containing the corresponding label for that face. For example, if the ith index in the list of faces represents the 4th individual in the database, then the corresponding ith location in the list of labels has value equal to 4. Now to perform the training using the Face Recognizer. Train function. It requires 2 arguments, the features which in this case are the images of faces and the corresponding labels assigned to these faces which in this case are the individual number that we extracted from the image names.

5.6	TESTING THE MODULE
For testing the Face Recognizer, we check if the recognition was correct by comparing the predicted label predicted with the actual label actual. The label actual is extracted using the OS module and the string operations from the name of the image. We also display the confidence score for each recognition.
CHAPTER NO. 6
TESTING
6.1	OVERVIEW
At the time of forming the dataset, each person will get designated using an id number. While recognition, when the test person image matches it will show the name of the person, if the test person image does not get matched with the dataset then no message will get send symbolizes a normal human being.
In this project, face detection is done using Viola- Jones algorithm and face extraction is done by LBPH algorithm as said earlier in this project. By following this process, there will be recognition rate accuracy of 95%.

6.2	TESTING ON DIFFERENT NUMBER OF FACES
Test #1: Single Face
In this project, we used two photos with a single face. However, in one photo we have the subject wearing shades. We wanted to test to see whether he can get a face detection even when the face was obscured by an object (the shades).
•	Parameters: scale Factor = 1.4
•	minNeighbors = 1
•	minSize = (30, 30)
Test #2: Two Faces
Next a photo with 2 faces showing different expressions. When there are more faces in an image, certain adjustments need to be made. Since the faces are smaller in the frame, we changed the minSize values until it was able to make the proper face detection. Even with one of the subjects smirking face showing a tongue, the faces were isolated and detected. The algorithm can still detect faces whether neutral or with emotions.
•	Parameters: scaleFactor = 1.4
•	minNeighbors = 1
•	minSize = (10,10)

Test #3: Three Faces
When we have more than 3 faces, things become more complicated. Now it requires having to set the parameters to a value that can identify the face correctly. In this subject are all lined up in the frame, so it would be much easier to locate and detect the faces.
•	Parameters: scaleFactor = 1.5
•	minNeighbors = 2
•	minSize = (30, 30)


CHAPTER NO. 7
CONCLUSION AND FUTURE WORK
Face Recognition is an imperative part of any industry. This work is most particularly for criminal identification, biometric authentication, games and multimedia entertainment. The algorithms carried out in this project were Viola-Jones algorithm and Linear binary pattern algorithm. The presented system will get implemented using Open CV. The recognition rate attained by this process is 90%-98%. There will be deviation in the result on account of the distance, camera resolution and lightning. Advanced processors can be put to use to reduce the processing time.
By affixing a greater number of recognition servers to attenuate the processing time for collection of images.









REFERENCES
1.	S L Suma, Sarika Raga. “Real Time Face Recognition of Human Faces by using LBPH and Viola Jones Algorithm.” International Journal of Scientific Research in Computer Science and Engineering, Vol.6, Issue.5, pp.01- 03, Oct. 2018.
2.	Li Cuimei, Qi Zhiliang. “Human face detection algorithm via Haar cascade classifier with three additional classifiers”, 13th IEEE International Conference on Electronic Measurement & Instruments, pp. 01-03, 2017.
3.	Senthamizh Selvi.R, D.Sivakumar, Sandhya.J.S, Siva Sowmiya.S, Ramya.S, Kanaga Suba Raja.S "Face Recognition Using Haar - Cascade Classifier for Criminal Identification" International Journal of Recent Technology and Engineering (IJRTE)ISSN: 2277-3878, Volume-7, Issue-6S5, April 2019.
4.	Varun Garg, Kritika Garg "Face Recognition Using Haar Cascade Classifier" December 2016, Volume 3, Issue 12.\
5.	“Face Detection Using Haar Cascade” https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection-using-haar-cascades.
6.	"Computer Vision — Detecting objects using Haar Cascade Classifier" https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9.

