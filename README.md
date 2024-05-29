# Handwritten Text Recognition System to Highlight Spelling Mistakes 

The goal of this project is to develop a system that accurately identifies and highlights spelling mistakes using handwritten text recognition(HTR) on the images. The code processes images by enhancing their quality, applies OCR to recognize handwritten text, and uses a spell-checking mechanism to detect spelling errors. Detected mistakes are then highlighted within the images, providing a clear visual indication of errors. This system aims to achieve high accuracy in both text recognition and error detection while ensuring clarity and efficiency in highlighting spelling mistakes.

## Image Preprocessing

The preprocessing steps are designed to enhance the quality of the handwritten text images to improve the accuracy of the OCR system. Here are the key preprocessing functions:

Resolution Enhancement (enhance_res): Increases the image resolution by a specified scale factor using cubic interpolation

Color Correction (color_corr): Converts the image to LAB color space and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the lightness channel.

Brightness and Contrast Adjustment (fix_bright_cont):Adjusts the brightness and contrast of the image using linear transformation

Edge Enhancement (edge_enhance): Converts the image to grayscale and applies Canny edge detection. The edges are then added back to the original image to enhance the sharpness and definition of the text.

Shadow Correction (fix_shadows): Splits the image into its RGB planes and removes shadows by subtracting the background using dilation and median blur techniques

Sharpening (fix_sharp): Applies a sharpening filter to the image

Image Enhancement (enhance_img): A combined function that applies all the above enhancement techniques to an image sequentially.

## Word Detection

The word detection part of the code is designed to locate and extract individual words from the preprocessed images. It is inspired by the scale space technique for word segmentation proposed by R. Manmatha and N. Srimal (1999). The method is fast, and achieves good accuracy with a relatively simple implementation.

Bounding Box Data Structure (BBox): Data class representing the bounding box coordinates of detected words in the image.

Detector Result Data Structure (DetectorRes): Data class that holds the cropped image of the detected word and its bounding box coordinates.

Detection Function (detect):
    1. Detects word regions in the image using an anisotropic filter and contour detection.
    2. Applies a filter kernel to enhance text features, converts the image to binary using Otsu's thresholding, and finds contours representing word boundaries.
    3. Filters out small contours based on a minimum area threshold and adds padding to each bounding box for better cropping.
    4. Returns a list of DetectorRes objects, each containing a cropped word image and its bounding box.

Kernel Computation (_compute_kernel): Computes the anisotropic filter kernel used in the detection process to enhance text features.

Image Preparation (prepare_img): Converts the image to grayscale (if needed) and resizes it to a specified height while maintaining the aspect ratio.

Clustering Lines (_cluster_lines):
    1. Groups detected word bounding boxes into lines based on their vertical proximity using the DBSCAN clustering algorithm.
    2. Uses a distance matrix based on the Jaccard distance to cluster words that are close together in the vertical axis.

Sorting Lines and Words (sort_multiline and sort_line): Sorts the lines of words based on their vertical position and within each line, sorts the words based on their horizontal position.

Saving Annotated Images (save_image_names_to_text_files): Saves the cropped word images and their names to a text file, and saves the annotated image with bounding boxes and labels for each word.

### Examples:

#### Original Images:

<img src="https://drive.google.com/uc?id=1sOPdylsFTdkCwhFEGIP8URVFLkKCxQXN" alt="alt text" width="400"/>

<img src="https://drive.google.com/uc?id=1EtcNIHTiRzlnt1Cd13HlyJ_tMAZIeEih" alt="blank 4" width="400"/>

<img src="https://drive.google.com/uc?id=1PRtO0C8zm0gir7Xt7P_8RELa5nW6He54" alt="IMG_1838" width="400"/>

<img src="https://drive.google.com/uc?id=16ODMCSY-RgXlcaKISLJ4bBkg6ls0NSEl" alt="blank 1" width="400"/>

<img src="https://drive.google.com/uc?id=1nJFAm_Ar9j6bgfCaqv_9MyePrIEPijkl" alt="blank 3" width="400"/>

#### Annotated Images:

<img src="https://drive.google.com/uc?id=1Qw0XnGkMYsTsmjoMAS1FXE2CievAVbhu" alt="download 1" width="400"/>

<img src="https://drive.google.com/uc?id=1ibjZswRetom7xtvHrExB6424B77aLbYp" alt="4" width="400"/>

<img src="https://drive.google.com/uc?id=1SAKXKnB5TbIjQAVN1qVrVctaCXCbMFPx" alt="2" width="400"/>

<img src="https://drive.google.com/uc?id=1qE693M2jDni9Gz1H_FMGQO5wg1_3xayz" alt="3" width="400"/>

<img src="https://drive.google.com/uc?id=1FPvLbdfVwMqMd5fXLhQEAwxSxKhcE5i5" alt="21" width="400"/>

### Parameter Tuning

The algorithm is not scale-invariant
    The default parameters give good results for a text height of 25-50 pixels
    If working with lines, resize the image to 50 pixels height
    If working with pages, resize the image so that the words have a height of 25-50 pixels
    The sigma parameter controls the width of the Gaussian function (standard deviation) along the x-direction. Small values might lead to multiply detection per word (over-segmentation), while large values might lead to a detection containing multiple words (under-segmentation)
    The kernel size depends on the sigma parameter and should be chosen large enough to contain as much of the non-zero kernel values as possible
    The average aspect ratio (width/height) of the words to be detected is a good initial guess for the theta parameter

<img src="https://drive.google.com/uc?id=15KAUHGElQ2t3BmHwc6-lC1Q_ZQIDzXp5" alt="download.png" width="400"/>

The filter kernel with size=33, sigma=28 and theta=10 is shown below on the left. It models the typical shape of a word, with the width larger than the height (in this case by a factor of 3). On the right the frequency response is shown (DFT of size 100x100). The filter is in fact a low-pass, with different cut-off frequencies in x and y direction.

## Preparing the Training Dataset

Since Google Colab and Google Drive can struggle with handling a large number of files and intensive I/O operations, I utilized the Kaggle API to download the IAM Handwriting Word Database. This dataset comprises a vast collection of labeled handwritten word images, which are essential for training and evaluating the OCR system.

The dataset is structured in the following manner, as explained in the words.txt file for the image metadata:

    #--- words.txt ---------------------------------------------------------------#
    #
    # iam database word information
    #
    # format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
    #
    #     a01-000u-00-00  -> word id for line 00 in form a01-000u
    #     ok              -> result of word segmentation
    #                            ok: word was correctly
    #                            er: segmentation of word can be bad
    #
    #     154             -> graylevel to binarize the line containing this word
    #     1               -> number of components for this word
    #     408 768 27 51   -> bounding box around this word in x,y,w,h format
    #     AT              -> the grammatical tag for this word, see the
    #                        file tagset.txt for an explanation
    #     A               -> the transcription for this word
    #
    a01-000u-00-00 ok 154 408 768 27 51 AT A


First, we define constants like batch size, image dimensions, and maximum text length are defined. 

    batch_size = 64
    image_width = 128
    image_height = 32
    max_len = 21


The images are then sorted using a natural key sorting method to ensure a logical order.
We also define a list of possible characters and save the character list to a file using the pickle module. This allows for the character set to be reused without redefining it every time the script runs. 

    characters = ['L', 'E', '"', '0', '2', 's', 'l', 'w', 'p', 'P', 'H', '/', 'O', 'V', 'f', '!', 'X', 'v', 'M', ':', 'N', '4', '3', "'", 'F', 'J', '(', 'K', '+', 'o', '#', 'I', 'r', '9', ',', '&', '8', 'B', '*', 'Q', 'g', '.', 'W', 'j', 'q', 'm', 'c', '5', ';', '7', 'U', 'h', 'u', 'a', ')', 'R', 'i', 'C', 'z', 'n', '-', 'Y', 'D', 'x', 'd', 'S', 'y', 'Z', 'e', 'k', 'G', '?', 'T', 'A', '6', 't', '1', 'b']

We can also remove all the numerical characters and punctuations here since our goal is setecting spelling errors.

Then, we apply the following preprocessing to the dataset to prepare it for training:

    resize_and_pad: resizes and pads images to ensure they have consistent dimensions, maintaining aspect ratio.

    preprocess_image: reads, processes, and saves images to the preprocessed folder, converting them to grayscale and normalizing pixel values. Then, stores the results in lists and checks if the images and labels were processed correctly.

    Character Mapping: extracts the unique characters from the labels, sorts them, and creates mappings from characters to numbers and vice versa using StringLookup.

    Vectorizing Labels: converts each label (string) into a tensor of integers using the character-to-number mapping and pads the sequences to a fixed length.

Examples from dataset:

<img src="https://drive.google.com/uc?id=17kZIJIEuYDMU1ZYXqtKMDxp6V69fRHoe" alt="a01-000u-00-00.png" width="400"/>
<img src="https://drive.google.com/uc?id=1CmzWdb1eVGhTHUgbAjcqyQVsPgrWZMq8" alt="a01-000u-00-01.png" width="400"/>
<img src="https://drive.google.com/uc?id=1HziUc9jO7_4yYai_HRLxYb0dI1xWdDtO" alt="a01-000u-00-02.png" width="400"/>
<img src="https://drive.google.com/uc?id=1T3-I_aYEwcq3DaWbInkKG4IWak9lu1vm" alt="a01-000u-00-03.png" width="400"/>
<img src="https://drive.google.com/uc?id=1V8TWFVz7FY_e6bYamZ8DT9TlZAZ584AB" alt="a01-000u-00-04.png" width="400"/>
<img src="https://drive.google.com/uc?id=1rWZY7tnnij56F7m0SjxR0arEPLBFPtsu" alt="a01-000u-00-05.png" width="400"/>
<img src="https://drive.google.com/uc?id=1t2eDVRAMrTPeKwckDIMxRRKrGLNLIL5B" alt="a01-000u-00-06.png" width="400"/>

## Model Architecture

Now, we define and construct the deep learning model architecture for recognizing handwritten text. The model combines convolutional neural networks (CNN) and recurrent neural networks (RNN) to effectively handle the spatial and sequential nature of the handwritten text images. Here's a breakdown of the key components and their purposes:

<img src="https://drive.google.com/uc?id=11HRE1Hyoyt9HT-niYYNou8te4sNNXcNE" alt="image-1.png" width="400"/>

Reference: [Handwritten Text Recognition Using TensorFlow 2.0](https://arthurflor23.medium.com/handwritten-text-recognition-using-tensorflow-2-0-f4352b7afe16)

### CTC Layer

    class CTCLayer(tf.keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.loss_fn = tf.keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            self.add_loss(loss)
            return y_pred

This class defines a custom Keras layer for the Connectionist Temporal Classification (CTC) loss, which is commonly used for sequence-to-sequence problems where the alignment between input and output is not known.

<img src="https://drive.google.com/uc?id=14nQdpFjbN0vHMErkCH_TZfT3b1uHb5iC" alt="image-2.png" width="400"/>

### CNN Layers

The input layer defines the input shape of the images.

    input_data = layers.Input(name='the_input', shape=(128, 64, 1), dtype='float32')

Then, we have 7 Convolutional Layers to extract spatial features from the input images. The convolutional layers are stacked with increasing filters to capture more complex features. Batch normalization and ReLU activation are applied after each convolution, followed by max-pooling to reduce the spatial dimensions.

After the convolutional layers, we have a reshape layer and a dense layer.

The reshape layer reshapes the output from the convolutional layers to a format suitable for the RNN layers.

    iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)


The dense layer adds a fully connected layer to further process the features before passing them to the RNN layers.

    iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

<img src="https://drive.google.com/uc?id=15E4AXaCI0ZqoY87oOewAQUkQvMq5AI2_" alt="image-3.png" width="400"/>

### RNN Layers

The output of these layers is passed to 2 RNN layers. Specifically, we are using Bidirectional GRU layers to process the sequence in both forward and backward directions and capture the sequential dependencies in the data.

    gru_1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(iam_layers)
    gru_1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(iam_layers)
    reversed_gru_1b = layers.Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

The forward and backward GRU outputs are merged and we then apply batch normalization.

    gru1_merged = layers.add([gru_1, reversed_gru_1b])
    gru1_merged = layers.BatchNormalization()(gru1_merged)

This process is repeated for the second RNN layer.

The final dense layer reduces the dimensionality to the number of possible characters, and the softmax activation outputs the probabilities of each character.

    iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(gru2_merged)
    iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

<img src="https://drive.google.com/uc?id=16-wmGvIM8LmM_sViJD5PuI7UYv71_owN" alt="image-4.png" width="400"/>

The model is finally compiled with the Adam optimizer, and the architecture is summarized below.

    __________________________________________________________________________________________________
    Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
    the_input (InputLayer)      [(None, 128, 64, 1)]         0         []                            
                                                                                                    
    conv1 (Conv2D)              (None, 128, 64, 64)          640       ['the_input[0][0]']           
                                                                                                    
    batch_normalization (Batch  (None, 128, 64, 64)          256       ['conv1[0][0]']               
    Normalization)                                                                                   
                                                                                                    
    activation (Activation)     (None, 128, 64, 64)          0         ['batch_normalization[0][0]'] 
                                                                                                    
    max1 (MaxPooling2D)         (None, 64, 32, 64)           0         ['activation[0][0]']          
                                                                                                    
    conv2 (Conv2D)              (None, 64, 32, 128)          73856     ['max1[0][0]']                
                                                                                                    
    batch_normalization_1 (Bat  (None, 64, 32, 128)          512       ['conv2[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_1 (Activation)   (None, 64, 32, 128)          0         ['batch_normalization_1[0][0]'
                                                                        ]                             
                                                                                                    
    max2 (MaxPooling2D)         (None, 32, 16, 128)          0         ['activation_1[0][0]']        
                                                                                                    
    conv3 (Conv2D)              (None, 32, 16, 256)          295168    ['max2[0][0]']                
                                                                                                    
    batch_normalization_2 (Bat  (None, 32, 16, 256)          1024      ['conv3[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_2 (Activation)   (None, 32, 16, 256)          0         ['batch_normalization_2[0][0]'
                                                                        ]                             
                                                                                                    
    conv4 (Conv2D)              (None, 32, 16, 256)          590080    ['activation_2[0][0]']        
                                                                                                    
    batch_normalization_3 (Bat  (None, 32, 16, 256)          1024      ['conv4[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_3 (Activation)   (None, 32, 16, 256)          0         ['batch_normalization_3[0][0]'
                                                                        ]                             
                                                                                                    
    max3 (MaxPooling2D)         (None, 32, 8, 256)           0         ['activation_3[0][0]']        
                                                                                                    
    conv5 (Conv2D)              (None, 32, 8, 512)           1180160   ['max3[0][0]']                
                                                                                                    
    batch_normalization_4 (Bat  (None, 32, 8, 512)           2048      ['conv5[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_4 (Activation)   (None, 32, 8, 512)           0         ['batch_normalization_4[0][0]'
                                                                        ]                             
                                                                                                    
    conv6 (Conv2D)              (None, 32, 8, 512)           2359808   ['activation_4[0][0]']        
                                                                                                    
    batch_normalization_5 (Bat  (None, 32, 8, 512)           2048      ['conv6[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_5 (Activation)   (None, 32, 8, 512)           0         ['batch_normalization_5[0][0]'
                                                                        ]                             
                                                                                                    
    max4 (MaxPooling2D)         (None, 32, 4, 512)           0         ['activation_5[0][0]']        
                                                                                                    
    conv7 (Conv2D)              (None, 32, 4, 512)           1049088   ['max4[0][0]']                
                                                                                                    
    batch_normalization_6 (Bat  (None, 32, 4, 512)           2048      ['conv7[0][0]']               
    chNormalization)                                                                                 
                                                                                                    
    activation_6 (Activation)   (None, 32, 4, 512)           0         ['batch_normalization_6[0][0]'
                                                                        ]                             
                                                                                                    
    reshape (Reshape)           (None, 32, 2048)             0         ['activation_6[0][0]']        
                                                                                                    
    dense1 (Dense)              (None, 32, 64)               131136    ['reshape[0][0]']             
                                                                                                    
    gru1_b (GRU)                (None, 32, 256)              247296    ['dense1[0][0]']              
                                                                                                    
    gru1 (GRU)                  (None, 32, 256)              247296    ['dense1[0][0]']              
                                                                                                    
    lambda (Lambda)             (None, 32, 256)              0         ['gru1_b[0][0]']              
                                                                                                    
    add (Add)                   (None, 32, 256)              0         ['gru1[0][0]',                
                                                                        'lambda[0][0]']              
                                                                                                    
    batch_normalization_7 (Bat  (None, 32, 256)              1024      ['add[0][0]']                 
    chNormalization)                                                                                 
                                                                                                    
    gru2_b (GRU)                (None, 32, 256)              394752    ['batch_normalization_7[0][0]'
                                                                        ]                             
                                                                                                    
    gru2 (GRU)                  (None, 32, 256)              394752    ['batch_normalization_7[0][0]'
                                                                        ]                             
                                                                                                    
    lambda_1 (Lambda)           (None, 32, 256)              0         ['gru2_b[0][0]']              
                                                                                                    
    concatenate (Concatenate)   (None, 32, 512)              0         ['gru2[0][0]',                
                                                                        'lambda_1[0][0]']            
                                                                                                    
    batch_normalization_8 (Bat  (None, 32, 512)              2048      ['concatenate[0][0]']         
    chNormalization)                                                                                 
                                                                                                    
    dense2 (Dense)              (None, 32, 80)               41040     ['batch_normalization_8[0][0]'
                                                                        ]                             
                                                                                                    
    softmax (Activation)        (None, 32, 80)               0         ['dense2[0][0]']              
                                                                                                    
    the_labels (InputLayer)     [(None, 16)]                 0         []                            
                                                                                                    
    input_length (InputLayer)   [(None, 1)]                  0         []                            
                                                                                                    
    label_length (InputLayer)   [(None, 1)]                  0         []                            
                                                                                                    
    ctc (Lambda)                (None, 1)                    0         ['softmax[0][0]',             
                                                                        'the_labels[0][0]',          
                                                                        'input_length[0][0]',        
                                                                        'label_length[0][0]']        
                                                                                                    
    ==================================================================================================
    Total params: 7017104 (26.77 MB)
    Trainable params: 7011088 (26.75 MB)
    Non-trainable params: 6016 (23.50 KB)
    __________________________________________________________________________________________________


#### Please note that due to the daily runtime limitations on google colab(especially gpu) and a time crunch, I was unable to train the model on the vast dataset properly. I will continue to try to train the model on my local machine and then use the saved model and weights to gather the inferences and refine the model for accuracy and efficiency using the vast dataset.

#### In order to complete the other tasks for the time being, I will proceed with a pretrained model for the time being.

## Word Recognition

First, we will load the cropped and processed images that were extracted by the word detection algorithm.

After loading preprocessed images from a specified directory, performing Optical Character Recognition (OCR) using PaddleOCR, and displaying the recognized text along with the probability scores for each text segment in the images.

We can see here how certain pictures/words can look ambiguous without context and how that can confuse the word recognition.

<img src="https://drive.google.com/uc?id=1JftvkZ8OcmJb8t57YWkFKhTANyKP-wDF" alt="image-5.png" width="400"/>
itself/itsey

<img src="https://drive.google.com/uc?id=1GK48pIUv6iNz9kB8rpWirizeQOZxzi7h" alt="download (11).png" width="400"/>
closed/dosed

Another issue is words of different lines getting joined due to letters like: f, g, j, q, y, p

<img src="https://drive.google.com/uc?id=1SeDtUDWjGp8vkBb0sYVhclmgwfQX0xSH" alt="image-6.png" width="400"/>

Recognized text: lisributed

## Marking Spelling Errors

We will use the text detection code from the start to processes images of handwritten text to detect text regions, and then perform Optical Character Recognition (OCR) on these regions, and highlight any misspelled words. 

For the spell-checking system, we are using SymSpell. For this, we have to download and load a frequency dictionary and a bigram dictionary. These dictionaries help the system identify and suggest correct spellings for detected words.

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = '/content/drive/MyDrive/HTR_Data/frequency_dictionary_en_82_765.txt'
    bigram_path = '/content/drive/MyDrive/HTR_Data/frequency_bigramdictionary_en_243_342.txt'

The detected text regions are clustered and sorted into lines of text using the DBSCAN algorithm. This helps in organizing the detected text for further processing.

    def _cluster_lines(detections: List[DetectorRes], max_dist: float = 0.7, min_words_per_line: int = 2) -> List[List[DetectorRes]]:
        num_bboxes = len(detections)
        dist_mat = np.ones((num_bboxes, num_bboxes))
        for i in range(num_bboxes):
            for j in range(i, num_bboxes):
                a = detections[i].bbox
                b = detections[j].bbox
                if a.y > b.y + b.h or b.y > a.y + a.h:
                    continue
                intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                union = a.h + b.h - intersection
                iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
                dist_mat[i, j] = dist_mat[j, i] = 1 - iou
        dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

        clustered = defaultdict(list)
        for i, cluster_id in enumerate(dbscan.labels_):
            if cluster_id == -1:
                continue
            clustered[cluster_id].append(detections[i])
        res = sorted(clustered.values(), key=lambda line: [det.bbox.y + det.bbox.h / 2 for det in line])
        return res

For each image, we perform OCR to extract text, then check the spelling of each detected word. Words identified as misspelled are highlighted in the image by drawing red rectangles around them.
The results, including the OCR output and the locations of misspelled words, are saved and displayed.

<img src="https://drive.google.com/uc?id=1zbmyQRgsIbEz6DRnoBu28iEArAy-qgAZ" alt="download (16).png" width="400"/>

Here, we can analyze the false positives as well to understand the reason for it.

The processed images with highlighted misspelled words are saved, and the recognized text results are displayed with their confidence scores and spelling status. For example:

    Word: Samurai, Confidence: 0.77, Misspelled: False
    Word: armi, Confidence: 0.97, Misspelled: True


## Evaluation

To evaluate the performance of the HTR system, we can calculate the two most common metrics: Word Error Rate (WER) and Character Error Rate (CER). 

The word_error_rate and character_error_rate functions use 'editdistance' to compute WER and CER for individual texts.

The decode_ocr_results function extracts and combines text from OCR results.

The evaluate_ocr function performs OCR on each test image, compares the OCR text with the ground truth text, and computes WER and CER.

For each image, the ground truth text, OCR result, WER, and CER are printed.

For images with decent handwriting we can expect the following average stats:

Word Error Rate (WER): 86.54%
Character Error Rate (CER): 47.48%

<img src="https://drive.google.com/uc?id=1X25Y8efPKFhLEdtv4OxdFG-qTs0HV5w3" alt="image-7.png" width="400"/>

<img src="https://drive.google.com/uc?id=1a7ZENy8IMM_v_nBZ6PYvr21XX_Lrptqb" alt="image-8.png" width="400"/>

Compared to images where even a human might struggle to understand the words without context, where we can expect average stats of:

Word Error Rate (WER): 94.35%
Character Error Rate (CER): 71.28%

<img src="https://drive.google.com/uc?id=1bgn0YUL9adXygIMIqjuJeoTP_Gl3USq7" alt="image-9.png" width="400"/>

<img src="https://drive.google.com/uc?id=1nhruNUDpsd66FEVXPp4FU6c4jziIRSvF" alt="image-10.png" width="400"/>

## Conclusion

The WER and CER are 86.54% and 47.48% for a reasonably well written page.

Even though these are not outstanding metrics, we can expect a better performance from a well-trained CRNN or Transformer based model. I will try to improve this soon.
