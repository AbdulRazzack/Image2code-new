# Image2code

##WEB PAGE CODE GENERATOR

(Image2Code)



                                                TEAM MEMBERS

  Syed Nazeeruddin Yousuf              : 20P71A6636
   Mohammmed Abdul Razzack       : 20P71A6602
                           Koyalkar Sai Kiran                        : 20P71A6654
                           Ragella Manish                              : 20P71A6630





##ABSTRACT


The project is based on generating code for the image of a web page given as an input by considering its UI ,frontend components and layouts as specified by the user.
    The code generator will automatically generate the code for the uploaded image. This code can be used by the industries for their frontend saving their precious time.

Project Objective:  
Delivering the suitable code for the web page design given as a photo to the Code Generator.
Specifying optimal and relevant code that runs with  approximate results.
Producing a responsive code for the web page image.
Maintaining time and space complexity of the code.

Project Drawbacks:
Misplacing of the components arises due to bad quality of the image.
It may be difficult for the code generator to characterize between the label ,button and links.
Improper orientation may be generated for the code due to blur images uploaded.
Code might not be generated for lame images or photographs which are not relevant for coding.
 If the page consists of many components in a single page it may take little extra time based on the processor the user is having.

Proposed project enhancement:
Imparting static and dynamic web page code for the specified image uploaded
Unlike the existing project which can only create code for most basic web pages , we try to go to the next level i.e a bit more complex.
Exhibiting features for the user to change from one component to another component before generating as the user can specify whether he wants a link or a button based on his requirement.
Making it responsive to run over any device with less number of bugs.




  


 





##Introduction

Code Generators : An Overview 

Code generators have evolved as transformative tools in the dynamic field of software development, automating the process of creating code from multiple sources. These technologies constitute a paradigm shift in software development, providing advantages that expedite development processes and improve code quality.

The capacity to automate repetitive and monotonous coding activities is at the heart of code generators, freeing up engineers to focus on more difficult and strategic parts of software development. These tools generate code that compiles to best practices and quality standards by leveraging established templates, rules, or requirements, ensuring uniformity and maintainability across the codebase.

Code generators are very important in shortening development times. These technologies enable developers to swiftly translate their ideas into functioning code by minimizing the need for manual coding.

Webpage Code Generators: 

A Specific generators' usefulness extends beyond standard programming languages, spanning a wide number of areas. These tools cater to numerous development demands, from producing database schema and data access code to designing user interfaces and web services, and provide a comprehensive solution for automating various elements of software creation.

As code generators progress, their impact on software development is expected to rise even more. These tools will get more sophisticated as artificial intelligence and machine learning improve, capable of understanding complicated specifications and writing code that is not only efficient but also optimized for performance and security.

Finally, code generators have transformed software development by streamlining workflows, increasing productivity, and boosting code quality. As these tools advance, they will likely play an ever more important role in determining the future of software development, enabling developers to design cus.

Within the broader domain of code generators, website code generators occupy a unique and valuable post. These specialized tools focus on automating the process of generating HTML and  CSS code for creating web pages, simplifying the process of translating design ideas into functional websites.
Webpage code generators address a common challenge faced by web developers: the time-consuming and error-prone task of converting visual designs into functional code. This manual coding process often involves repetitive tasks and can hinder the overall development speed, especially for those with limited coding expertise.

To overcome these challenges, webpage code generators provide user-friendly interfaces that enable designers and developers to specify the desired layout, content, and interactive elements of a web page. The tool then analyzes these specifications and generates the corresponding HTML and  CSS.

This automated approach offers several advantages:

1. Enhanced Productivity: By eliminating the need for manual coding, webpage code generators significantly enhance developer productivity, allowing them to focus on more creative and strategic aspects of web development.
2. Reduced Development Time: The ability to quickly generate code from design specifications accelerates the development process, enabling developers to bring websites to life more efficiently.
3. Improved Code Quality: Webpage code generators ensure that the generated code adheres to best practices and quality standards, leading to more maintainable and reliable websites.
4. Accessibility for Non-Coders: These tools empower individuals with limited coding expertise to create web pages, democratizing web development and expanding the pool of website creators.
5. Consistency in Design and Code: Webpage code generators promote consistency in both design and code, ensuring that the visual appearance of the website aligns with the underlying code structure.
6. Rapid Prototyping and Iteration: The ability to quickly generate code from design ideas facilitates rapid prototyping and iteration, enabling designers and developers to refine and improve website designs more effectively.
7. Integration with Existing Projects: The generated code can be easily integrated into existing web development projects, providing flexibility and customization options.
8. Cross-Platform Compatibility: Webpage code generators often produce code that is compatible with various browsers and devices, ensuring that websites display correctly across different platforms.
In conclusion, website code generators have emerged as valuable tools that streamline the process of creating web pages, enhancing productivity, reducing development time, and improving code quality. By automating the conversion of design ideas into functional code, these tools empower a wider range of individuals to create visually appealing and technically sound websites. As webpage code generators continue to evolve, their impact on web development is poised to grow even further, shaping the future of website creation and enabling a more accessible and efficient development process.

Projects Motivation and Purpose:

Automating Web page Creation. The Project Webpage Image Code Generator stems from the desire to address the challenges faced in manually converting webpage designs into functional code. This manual process is often time-consuming, error-prone, and requires a certain level of coding expertise, hindering the overall development speed and limiting the accessibility of web development for those with limited coding experience.

The project aims to provide an automated solution for generating web page code from images, enabling users to effortlessly create web pages simply by uploading an image depicting the desired layout and content. This approach offers several advantages:

Effortless Web page Creation: By eliminating the need for manual coding, the tool empowers users to create web pages without the need for coding expertise, expanding the pool of website creators and democratizing web development.
Seamless Design-to-Code Conversion: The ability to directly generate code from design images streamlines the development process, allowing designers and developers to focus on the creative aspects of website creation.
Enhanced Productivity: Automating the code generation process significantly reduces development time, enabling developers to bring websites to life more efficiently and focus on more complex and strategic aspects of web development.
Improved Code Quality: The tool ensures that the generated code adheres to best practices and quality standards, leading to more maintainable and reliable websites.
Rapid Prototyping and Iteration: The ability to quickly generate code from design images facilitates rapid prototyping and iteration, enabling designers and developers to refine and improve website designs more effectively.
Accessibility for Non-Coders: The tool empowers individuals with limited coding expertise to create web pages, breaking down barriers to entry and expanding web development opportunities.

In essence, the Project Image2Code - Webpage Image Code Generator aims to revolutionize the way web pages are created, transforming the process from a tedious and technical endeavor into a seamless and accessible experience. By automating design-to-code conversion, the tool empowers individuals to bring their creative ideas to life and contribute to the dynamic world of web development.







##LITERATURE SURVEY

To identify areas necessitating further research in the context of our project, an examination of related work and research papers has been conducted. Noteworthy studies by Ali Davody, Homa Davoudi, Mihai S. Baba, R˘azvan V. Florian reveal that current learning-based program synthesis methods often rely on ground-truth programs generated by programmers. However, this reliance on human-generated ground-truth may limit the efficiency and flexibility of programming models, especially in scenarios where different programmers may approach problem-solving with distinct algorithms, structures, operations, and variable names.

In light of this insight, we draw inspiration from automated HTML code generation using an object detection model, specifically the YOLO (You Only Look Once) model, as a potential solution to the challenge of program synthesis. Recent endeavors, particularly those employing machine learning techniques, including deep neural networks for program synthesis, have shown promise. Our focus is on utilizing the YOLO model for object detection in images, and subsequently generating HTML templates based on the detected objects.

In our proposed model, the YOLO model acts as the object detection system. It takes a webpage image as input and identifies objects within the image. Subsequently, an HTML template is generated based on the detected objects. To train the model, we employ a supervised learning approach, where the model learns to detect objects and generate HTML templates concurrently. The reward signal is computed by assessing the accuracy of the generated HTML template in rendering the detected objects accurately.

During the testing phase, the trained YOLO model can process unseen images, detect objects, and generate corresponding HTML templates using the learned policy network. Importantly, our approach allows the model to adapt specifically to the test image, presenting an opportunity for further improvement. This adaptability is facilitated by continued learning during the testing phase, leveraging the reward signal provided by assessing the accuracy of the generated HTML template in rendering the detected objects. This process can continue until the model achieves optimal performance.

In the context of our project involving an augmented image dataset with multi-class annotations, this approach holds promise for advancing the synthesis of HTML templates from visual inputs, offering potential applications in the field of computer vision and image analysis, especially in scenarios where diverse object types need to be accurately represented in generated HTML templates.



##Analysis 

Analysis and Feasibility Report:

The  project's feasibility was assessed by examining the technical requirements and the capabilities of existing machine learning frameworks to handle image-to-code translation tasks. The assessment involved evaluating the complexity of user interface (UI) designs that could be translated into code and the adaptability of neural network architectures for this purpose.

Analysis of the Model:

The model is a custom trained YOLOv8 architecture model. YOLO, or a "You Only Look Once," is an object identification architecture that operates on a single stage and uses the input image to forecast bounding boxes and class probabilities. While YOLO can be faster than two-stage object detection models, it can also be less accurate.  This is used for extracting features from the input images, while the extracted elements are used to  generate the corresponding code. The analysis covered the model's ability to capture both the structural and stylistic nuances of UI designs.
 
Challenges in Developing a Code Generator Model:

Key challenges identified include:

Diversity of UI Components:
            UIs can be incredibly varied, with different styles, layouts, and interactive elements.      
            The model must generalize well across this diversity.
Accuracy of Code Generation:
The generated code must not only replicate the visual aspect of the UI but also adhere to syntactic correctness and functional soundness.
Contextual Understanding:
The model must understand the context of different UI elements, such as a button being part of a form, to generate semantically correct code.
Performance Optimization:
Processing images and generating code in real-time demands high computational efficiency and optimization of the model's performance.

Assumptions and Dependencies:

The project assumes that:
The input images are of high quality and represent the UI clearly.
There are enough examples of different UI components within the training data to learn from.
The dependencies include the availability of robust machine learning libraries and sufficient computational resources to train complex models.

Constraints and Limitations:

The project faces constraints such as:

Computational Resources: Training deep learning models is resource-intensive, requiring powerful hardware.
Quality of Training Data: The accuracy of the model is heavily dependent on the quality and variety of the training dataset.
Model Complexity: There is a trade-off between the model's complexity and its ability to generalize across unseen UI designs.

Risks and Mitigation Strategies:

Risks include:

Overfitting: The model may overfit to the training data, impairing its ability to generalize. This can be mitigated by using regularization techniques and expanding the dataset.
Underperforming Model: If the model fails to achieve the desired accuracy, alternative architectures or additional features could be explored.
Technological Advances: The model might need to be updated frequently due to the rapid developments in web technologies. To reduce this risk, stay up to date on industry developments and keep the model design flexible.
                                                                                      




##Project Requirements

Software Requirements

Programming Languages:

Python: The primary language used for scripting, data manipulation, and model development.

HTML/CSS (for generated output): The target languages for the generated code from the system.

Frameworks and Libraries:

Flask:
Flask is a lightweight and versatile web framework for Python. It facilitates the development of web applications and RESTful APIs, providing tools for URL routing, request handling, and template rendering. Flask's simplicity and extensibility make it a popular choice for building web services.
Flask-CORS:
Flask-CORS is an extension for Flask that simplifies Cross-Origin Resource Sharing (CORS) in web applications. It allows for secure communication between a web application served from one origin and resources from another origin, addressing browser security policies that restrict such interactions.
Torch:
Torch is an open-source machine learning library that provides a flexible, efficient, and dynamic computational graph. It is widely used for deep learning tasks, offering a comprehensive set of tools for building and training neural networks. PyTorch, a popular deep learning framework, is built on top of Torch.
Ultralytics:
Ultralytics is a deep learning research platform that focuses on computer vision and object detection tasks. It provides easy-to-use APIs and pre-configured models, making it efficient for developing, training, and deploying state-of-the-art models. YOLOv5, a real-time object detection model, is among the notable projects within Ultralytics.
Werkzeug:
Werkzeug is a WSGI (Web Server Gateway Interface) utility library for Python. It is commonly used as part of web frameworks, including Flask. Werkzeug provides essential components for handling HTTP requests, routing, and other web-related functionalities.
Pillow:
Pillow is a powerful Python Imaging Library fork that adds support for opening, manipulating, and saving various image file formats. It serves as a versatile tool for image processing and is commonly used in web applications for tasks such as image resizing, cropping, and filtering.
NumPy:
NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. NumPy is widely used in various scientific and machine learning applications, forming the foundation for many numerical operations.

Development Tools:

Git: For version control, allowing for efficient tracking of changes and collaboration.
GitHub: The code repository hosting service used for the project's source code and documentation.
Integrated Development Environment (IDE): IDE’s  such as PyCharm or Visual Studio Code: To provide a robust environment for writing, testing, and debugging code.

Data Serialization:

JSON: To store and exchange the output elements detected by the model and for generating the code.

Hardware Requirements:

Processing Power:
CPU with multiple cores for efficient parallel processing during the model training phase.
GPU support (preferably with CUDA compatibility) to accelerate the training of deep learning models.
Memory:
Sufficient RAM (at least 16GB recommended) to handle the large datasets and the resource-intensive training processes.
High-speed SSD with ample storage to facilitate quick read/write operations for the dataset and model checkpoints.

Functional Requirements:

Input/Output:
The system must accept GUI images as input and output the corresponding HTML/CSS code.

The input images should be in a common format such as JPEG or PNG.

The output code must be compatible with standard web browsers and comply with web standards.

Image Dataset Management:

Upload and Storage: Allow users to upload images to the system and store them securely.
Augmentation: Implement image augmentation processes to increase the diversity of the dataset.
Annotation Management: Enable the addition, editing, and deletion of annotations associated with each image.
Define Classes: Provide the ability to define and manage annotation classes such as "header," "footer," "card," "image," and "text."
Assign Classes: Allow users to assign relevant classes to specific regions or elements within each image.

Data Retrieval and Display:

Search and Filter: Implement a search and filter mechanism to retrieve images based on classes, annotations, or other criteria.
Display Annotations: Visualize annotated images, highlighting different classes and regions within each image.

Machine Learning Integration (if applicable):

Training Data Preparation: Facilitate the extraction of training data for machine learning models, ensuring compatibility with frameworks like Torch and Ultralytics.
Inference Support: If applicable, integrate features for model inference on new, unseen data.

Web Application Features (if using Flask and related tools):

User Authentication: Implement user authentication and authorization mechanisms to control access to the dataset and functionalities.
API Endpoints: Develop RESTful API endpoints for managing and retrieving data.
Cross-Origin Resource Sharing (CORS): Utilize Flask-CORS to handle cross-origin requests if the application interacts with other web services.

Image Processing (if using Pillow and NumPy):

Image Manipulation: Provide tools for basic image processing operations such as resizing, cropping, and filtering using Pillow and NumPy.

Documentation and Help:

User Guides: Develop comprehensive documentation for users, explaining how to use the system, manage annotations, and integrate with machine learning frameworks.
Support and Help Desk: Offer a mechanism for users to seek assistance, report issues, or ask questions.

Scalability and Performance:

Scalability: Design the system to handle a growing number of images and annotations efficiently.
Performance Optimization: Implement optimizations to ensure fast retrieval and display of annotated images.

Security:

Data Security: Ensure the security of uploaded images and annotations.
Authentication and Authorization: Implement secure user authentication and authorization mechanisms.

Testing:

Unit Testing: Conduct unit tests to verify the functionality of individual components.
Integration Testing: Perform integration tests to ensure seamless interaction between different modules.

Non - Functional Requirements

Performance:

Response Time: Specify the maximum acceptable response time for queries and operations within the system.
Throughput: Define the number of transactions or requests the system should handle per unit of time.

Scalability:

Capacity Planning: Identify the expected growth in the dataset and user base, and ensure the system can scale accordingly.
Load Balancing: Implement mechanisms for distributing workload efficiently across servers to handle increased demand.

Reliability:

Availability: Specify the desired level of system availability, including planned downtime for maintenance.
Fault Tolerance: Define how the system should behave in the face of failures, ensuring data integrity and minimal service disruption.

Security:

Data Encryption: Specify requirements for encrypting sensitive data during storage and transmission.
Access Control: Define access levels and permissions for different user roles, ensuring data confidentiality and integrity.

Usability:

User Interface Design: Define standards for the user interface design to ensure a consistent and intuitive user experience.
Accessibility: Ensure that the application is accessible to users with disabilities.

Maintainability:

Code Maintainability: Specify coding standards and practices to ensure code is readable, well-documented, and easy to maintain.
Modularity: Design the system with a modular architecture to facilitate updates and enhancements.


Compatibility:
Browser Compatibility: Specify the browsers and versions that the web application should support.
Framework Compatibility: Ensure compatibility with specific versions of frameworks and libraries, such as Flask and Torch.

Documentation:

Technical Documentation: Provide comprehensive technical documentation for developers, including code documentation, APIs, and system architecture.
User Documentation: Offer user-friendly guides for end-users explaining how to interact with the system.

Performance Testing:

Load Testing: Conduct load testing to ensure the system can handle the expected volume of concurrent users and transactions.
Stress Testing: Assess the system's robustness by subjecting it to stress conditions beyond normal operational levels.

Compliance:

Legal and Regulatory Compliance: Ensure that the system complies with relevant laws, regulations, and industry standards regarding data privacy and security.

Monitoring and Logging:

Logging Requirements: Define the level of detail required in system logs for auditing, debugging, and monitoring.
Monitoring Tools: Implement tools for monitoring system performance and detecting anomalies.






##Design and Architecture


Fig 1:convolution layers for imageNet classification 

Design and Lifestyle:

The general design of the project is delivering the suitable code for the web page design given as a photo to the Code Generator. It specifies optimal and relevant code that runs with  approximate results. It produces a relative code for the web page image.

Proposed Process Model: 

Incremental Model:

The incremental model is a software development model where the development process is divided into smaller, more manageable pieces or increments. Each increment builds upon the functionality of the previous increment and is delivered to the customer for feedback and testing. This approach allows for the incremental delivery of a working system, with each increment being more complete than the previous one.

The incremental model is particularly useful when the requirements are not well understood or are likely to change, as it allows for flexibility and adaptability during the development process. It also allows for early feedback and testing from the customer, which can help to identify and address issues early in the development process.


Architecture:
At the core of our system is the AI model of course, that is doing the actual work. The process begins with the YOLO model, a specialized neural network that takes an image as input and  then identifies or detects the elements from the given input image. After theYOLO model, the input advances towards' a sophisticated structure where the actual translation occurs i,e stripping away the redundant information from the results object while preserving the core structure necessary for code generation. By integrating the insights from the YOLO model with the generated results and understanding the correlation between visual elements and their corresponding code structures. The beauty of this design lies in its simple yet concise structure . While many tend to focus on complex approaches, the main model takes into account simple steps that bridges the gap between graphic design and HTML code. This approach ensures a comprehensive understanding of the task at hand, leading to remarkably accurate code generation.

The YOLO model

Fig 2: The yolo model for object detection




Fig 3:


Fig 3: The YOLOv6 framework (N and S are shown).

Fig 4: Bondings and confidence level for object detection



##Convolutional Neural Network:

A Convolutional Neural Network (CNN) is a type of Deep Learning neural network architecture commonly used in Computer Vision. Computer vision is a field of Artificial Intelligence that enables a computer to understand and interpret the image or visual data. 
When it comes to Machine Learning, Artificial Neural Networks perform really well. Neural Networks are used in various datasets like images, audio, and text. Different types of Neural Networks are used for different purposes, for example for predicting the sequence of words we use Recurrent Neural Networks more precisely an LSTM, similarly for image classification we use Convolution Neural networks.

In a regular Neural Network there are three types of layers:

Input Layers: 
It’s the layer in which we give input to our model. The number of neurons in this layer is equal to the total number of features in our data (number of pixels in the case of an image).
Hidden Layer:
The input from the Input layer is then fed into the hidden layer. There can be many hidden layers depending on our model and data size. Each hidden layer can have different numbers of neurons which are generally greater than the number of features. The output from each layer is computed by matrix multiplication of the output of the previous layer with learnable weights of that layer and then by the addition of learnable biases followed by activation function which makes the network nonlinear.
Output Layer:
The output from the hidden layer is then fed into a logistic function like sigmoid or softmax which converts the output of each class into the probability score of each class.The data is fed into the model and output from each layer is obtained from the above step is called feedforward, we then calculate the error using an error function, some common error functions are cross-entropy, square loss error, etc. The error function measures how well the network is performing. After that, we backpropagate into the model by calculating the derivatives. This step is called Backpropagation which basically is used to minimize the loss.

Convolution Neural Network
Convolutional Neural Network (CNN) is the extended version of artificial neural networks (ANN) which is predominantly used to extract the feature from the grid-like matrix dataset. For example visual datasets like images or videos where data patterns play an extensive role.

CNN architecture

Convolutional Neural Network consists of multiple layers like the input layer, Convolutional layer, Pooling layer, and fully connected layers

Fig 5: Layers of CNN architecture
The Convolutional layer applies filters to the input image to extract features, the Pooling layer downsamples the image to reduce computation, and the fully connected layer makes the final prediction. The network learns the optimal filters through backpropagation and gradient descent.

How Convolutional Layers works

Convolutional Neural Networks or covnets are neural networks that share their parameters. Imagine you have an image. It can be represented as a cuboid having its length, width (dimension of the image), and height (i.e the channel as images generally have red, green, and blue channels). 

Fig 6: image depicting convolutional layers works
Now imagine taking a small patch of this image and running a small neural network, called a filter or kernel on it, with say, K outputs and representing them vertically. Now slide that neural network across the whole image, as a result, we will get another image with different widths, heights, and depths. Instead of just R, G, and B channels now we have more channels but lesser width and height. This operation is called Convolution. If the patch size is the same as that of the image it will be a regular neural network. Because of this small patch, we have fewer weights. 


Fig 7: Running a covnets on an image of dimension
Layers used to build ConvNets
A complete Convolution Neural Networks architecture is also known as covnets. A covnets is a sequence of layers, and every layer transforms one volume to another through a differentiable function. 
Types of layers: datasets
Let’s take an example by running a covnets on an image of dimension 32 x 32 x 3. 

Input Layers: It’s the layer in which we give input to our model. In CNN, Generally, the input will be an image or a sequence of images. This layer holds the raw input of the image with width 32, height 32, and depth 3.
Convolutional Layers: This is the layer, which is used to extract the feature from the input dataset. It applies a set of learnable filters known as the kernels to the input images. The filters/kernels are smaller matrices, usually 2×2, 3×3, or 5×5 shape. It slides over the input image data and computes the dot product between kernel weight and the corresponding input image patch. The output of this layer is referred to as feature maps. Suppose we use a total of 12 filters for this layer we’ll get an output volume of dimension 32 x 32 x 12.
Activation Layer: By adding an activation function to the output of the preceding layer, activation layers add nonlinearity to the network. it will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are RELU: max(0, x),  Tanh, Leaky RELU, etc. The volume remains unchanged hence output volume will have dimensions 32 x 32 x 12.
Pooling layer: This layer is periodically inserted in the covnets and its main function is to reduce the size of volume which makes the computation fast, reduces memory and also prevents overfitting. Two common types of pooling layers are max pooling and average pooling. If we use a max pool with 2 x 2 filters and stride 2, the resultant volume will be of dimension 16x16x12. 

Fig 8:pooling layer that reduces the memory by slicing it and preventing overfitting
Flattening: The resulting feature maps are flattened into a one-dimensional vector after the convolution and pooling layers so they can be passed into a completely linked layer for categorization or regression.
Fully Connected Layers: It takes the input from the previous layer and computes the final classification or regression task.












##Data and Dataset
Data Description:

This dataset comprises a diverse collection of 249 images, initially starting with 119 images that underwent augmentation to enhance the variety and richness of the dataset. The images within this dataset have been meticulously annotated, providing valuable insights into the diverse elements present in each image.

The annotations encompass a range of classes, allowing for a comprehensive understanding of the image content. These classes include, but are not limited to, "header," "footer," "card," "image," and "text." The meticulous annotation process ensures that each element within the images is accurately labeled, enabling the development and training of machine learning models for tasks such as object recognition, segmentation, and classification.

Researchers, developers, and machine learning practitioners can leverage this dataset to advance their work in computer vision and image analysis. The augmented nature of the dataset enhances its utility by exposing models to a broader set of scenarios and variations, contributing to improved robustness and performance.

Whether you are working on image classification, object detection, or other computer vision tasks, this annotated and augmented dataset serves as a valuable resource for training and testing models in a variety of real-world scenarios. The detailed annotations provide a foundation for the development of models capable of recognizing and categorizing diverse visual elements within images, making it an essential asset for advancing the field of computer vision.

The Dataset is a custom dataset, prepared by the team, where many website images and then classes are individually annotated.

PNG, JPG : PNG, JPG files is the main design image of the HTML which will be used to generate the HTML Design Code.
HTML Code : These files consist of the HTML code of the given image.
Summary :
Input :*.PNG, *JPG etc files
Output :-> HTML Code file 

Sample Image and its GUI representation in dataset:


Fig 9: Sample image representing GUI dataset

Fig 10: Yolo model detecting the objects present in sample image

Source of Information: The Image2Code project makes use of a custom dataset that includes images of graphical user interfaces (GUIs). These graphics are made synthetically to illustrate various layouts and components used in development.

Images of various GUI layouts, such as buttons, text, and other typical design elements, are included in the dataset. These photos are fed into the model, which attempts to identify the web ui element. 

Preparation and Preprocessing of Data:

Data Loading:
Load the augmented dataset into your project environment, ensuring access to both image data and corresponding annotations.
Splitting the Data:
 Separate the dataset into training, testing, and validation sets based on the specified 
             percentages. 
Image Resizing and Normalization:
Resize all images to a consistent size suitable for model input.(640x640)
            Normalize pixel values to a common scale (e.g., [0, 1] or [-1, 1]) to facilitate         
            convergence during training.
Annotation Processing:
Ensure that annotations are aligned with the resized images.Convert annotation data into a format compatible with your chosen machine learning framework (e.g., bounding box coordinates, class labels).
Data Balancing:
If your dataset has imbalances in class distribution, consider techniques to balance the   
            classes in the training set, such as oversampling or undersampling.
Data Split Validation:
Verify that the data split ratios align with the initial distribution goals (70% training, 20% testing, 10% validation).
Data Storage and Versioning:
Establish a system for storing and versioning your prepared datasets, ensuring traceability and reproducibility.
Quality Checks:
Perform quality checks on the data, including ensuring the correct alignment of images and annotations, handling missing or corrupted data, and addressing any anomalies.

##Data Split for Training, Testing and Validation:

The process of splitting your dataset into training, testing, and validation sets is a critical step in machine learning model development. The goal is to have distinct subsets for training the model, evaluating its performance during development, and assessing its generalization on unseen data. A common split ratio is 70% for training, 20% for testing, and 10% for validation.
Division Strategy: To guarantee that the model learns what it needs to, the dataset is divided into training, testing and validation sets, with a sizable chunk set aside for training.

Training set:
The training set is the most important set of data in the machine learning process. It is used to train the model to perform the desired task. The training set should be as large and diverse as possible to ensure that the model is able to learn from a wide range of data.
Testing set:
The testing set is used to evaluate the performance of the model on unseen data. This is important to ensure that the model is not overfitting to the training data. The testing set should be representative of the data that the model will be used on in production.
Validation set
The validation set is used to tune the hyperparameters of the model. Hyperparameters are the settings that control the training process, such as the learning rate and the number of epochs. The validation set is used to select the values of the hyperparameters that produce the best performance on the validation set.
























Supervised Learning

Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. Supervised learning helps organizations solve a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

How does Supervised Learning works

Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.
Supervised learning can be separated into two types of problems when data mining—classification and regression:
Classification uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. 
Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.
Regression is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistic regression, and polynomial regression are popular regression algorithms.
Usage of Supervised learning:
Supervised learning has a wide range of use cases across various domains. Here are some common use cases of supervised learning:
1. Convolutional Neural Networks (CNNs):
Image Classification:
CNNs are widely used for image classification tasks. Supervised learning involves training a CNN on a labeled dataset of images with corresponding class labels.The model learns to extract hierarchical features from the images, enabling it to recognize and classify objects.
Transfer Learning:
Pre-trained CNN models (e.g., VGG, ResNet, Inception) on large datasets like ImageNet are often used for transfer learning.Fine-tuning the pre-trained models on a specific object detection dataset with additional labeled bounding box information is a form of supervised learning.
2. YOLO (You Only Look Once) Model:
Object Detection:
YOLO is a popular real-time object detection model that divides an image into a grid and predicts bounding boxes and class probabilities for objects within each grid cell.Supervised learning is employed to train YOLO on datasets with images annotated with bounding boxes and class labels for each object of interest.
Loss Function:
YOLO utilizes a specific loss function that combines classification loss (objectness and class probabilities) and regression loss (bounding box coordinates) for each grid cell.The loss is minimized during training to improve the accuracy of object localization and classification.
Multiple Object Detection:
YOLO's strength lies in its ability to detect multiple objects in an image simultaneously. Supervised learning helps the model generalize to various object classes and scenarios.
Real-time Applications:
YOLO is suitable for real-time applications, such as video surveillance, autonomous vehicles, and robotics, where quick and accurate object detection is crucial.
In both CNNs and YOLO, the supervised learning process involves feeding labeled training data to the model, calculating a loss based on the model's predictions compared to ground truth, and iteratively adjusting the model's parameters to minimize the loss. The trained models can then make accurate predictions on new, unseen data for image classification (CNN) or object detection (YOLO).
These use cases demonstrate the versatility of supervised learning in solving a wide array of problems by training models on labeled datasets to make predictions or classifications on new, unseen data.
CNN:
CNNs are particularly effective for image-related tasks due to their ability to automatically learn hierarchical features from the data. The convolutional layers capture patterns and features at different scales, enabling the network to recognize complex patterns in images. Supervised learning with CNNs has been successful in various applications, including image classification (assigning a label to an entire image), object detection (identifying and locating objects within an image), and image segmentation (assigning labels to individual pixels or regions in an image).
Model Architecture:Choose a CNN architecture suitable for object detection, such as Faster R-CNN, YOLO, or SSD.
Data Preprocessing:Resize input images to a fixed size.Normalize bounding box coordinates.
Loss Function:Define a loss function that combines classification loss (for predicting object classes) and regression loss (for predicting bounding box coordinates).
Training:Train the model on the labeled dataset by feeding images through the network and adjusting parameters based on backpropagation and optimization.Use iterative training, monitoring performance on a validation set.
Validation:Evaluate the model on a separate validation set to fine-tune hyperparameters and prevent overfitting.
Testing:Test the trained model on a separate test dataset to assess its generalization to new, unseen data.
Inference:Deploy the trained model for inference on new images.Obtain predictions for object classes and bounding box coordinates.
Post-processing:Apply post-processing techniques like non-maximum suppression to filter redundant or overlapping bounding box predictions.
Fine-tuning (Optional):Optionally, use transfer learning by initializing the CNN with pre-trained weights from a model trained on a large dataset.
Objective: The process involves preparing a labeled dataset, selecting a suitable CNN architecture, defining a loss function, training the model, validating its performance, testing on unseen data, and deploying for inference. Post-processing techniques are applied to refine predictions, and fine-tuning can be considered for transfer learning. This approach allows the model to learn to detect and classify objects in images accurately.

Machine Learning Techniques 

Introduction to Machine Learning in GUI Code Generation:

Relevance to Project: The project leverages machine learning (ML) techniques, particularly deep learning, to translate GUI images into code. This approach is central to the project, enabling automated code generation from visual inputs.

Choice of Machine Learning Model

Convolutional Neural Networks (CNNs):

 CNNs are used for their ability to effectively process and interpret image data. In this context , CNNs analyze GUI images, identifying patterns and features relevant to code generation.
Sequence-to-Sequence Model: The project employs a sequence-to-sequence model, which is crucial for translating the processed image data (input sequences) into corresponding code (output sequences). This model type is adept at handling such sequential data transformation tasks.

Implementation of the ML Model

Model Architecture: 

The architecture includes an object detection model i.e YOLO. The model processes the input GUI images, and then generates the result object consisting of the elements and their corresponding coordinates. These are then used to generate code for the image.

Training the Model: 

The model is trained on the dataset of GUI images which are annotated on the mentioned classes and it learns to detect web ui elements from the image and use it to generate code snippets.

Loss Function and Optimizers:

The model uses specific loss functions and optimizers to refine its learning process, focusing on minimizing the discrepancy between the generated code and the actual code.

Usage of ML Techniques in the Project:

Automated Code Generation: The core application of ML in  is to automate the process of generating functional code from GUI images, a task traditionally done manually by developers.
Handling Variability: ML techniques enable the model to handle a wide range of GUI designs, making the tool versatile and adaptable to various design inputs.
            Neural Networks and Layers Description.
Layers in CNN: The CNN comprises multiple layers, including convolutional layers, pooling layers, and fully connected layers, each playing a role in feature extraction and interpretation.























##Algorithm and Flowchart of the Proposed Model

Process Flowchart



Fig 11: A flowchart that illustrates the step-by-step how the model follows, from data input to generating HTML and CSS code.








The Elements of the Architecture:

Input Image: This is the starting point of the process, where a GUI image is provided as input to the trained AI model. The image contains visual elements of a user interface that need to be translated into code.
Trained AI Model: The AI model has been previously trained on a dataset of images and. It uses this training to interpret the input image and generate a corresponding results object which contains the coordinates of the detected object and their class names. The training enables the model to recognize patterns and structures in the GUI that are common to web design.
Code Generation Script: This component takes the results object generated by the YOLO model and uses the code generation script to convert it into HTML code. It understands the syntax of the webpage and translates it into a more conventional programming language that can be rendered by web browsers.
HTML Code: The final output of the process is the HTML code, which is the standard code used to create web pages. This code can be directly used in web development, allowing the design from the input image to be rendered as a functional web interface.

Algorithm of the Model :

A detailed algorithm for image-to-code conversion model.

The Backend Architecture:

Input Image: This is the graphical user interface (GUI) image that is fed into the system. It represents a user interface design with their corresponding annotations of classes, from which code needs to be generated.
YOLO:  It is a real-time object detection model that utilizes a CNN to identify and locate objects. It employs a unique single-stage approach for real-time processing. The model's backbone extracts features, and the neck network refines them. The head performs detection and classification.
Web UI Element Detection: The YOLO model compresses the input image into a lower-dimensional representation, capturing the essential features needed for detecting the elements and getting  the  web ui elements from the image.

Fig 12: web UI code generation flowchart

Result Object : The  Results object which is generated from the YOLO models prediction contains the coordinates of the detected elements from the given image. These coordinates are then  used in an attempt  to generate the code  from this json representation of the de. This step is crucial to ensure that the essential features or classes present in  the original image are accurately predicted and represented in the results object . The quality of the predicted result object is often an indicator of the effectiveness of the feature extraction process.

Code Generation : A starting point for the main essence of the project is to begin generating code, which takes input based on the features extracted from the input image.
Main Model: This is the central component of the system, responsible for generating the final code output. The input is:
The feature representation from the results object of the YOLO model, which has the essential visual information from the input image .
            The main model uses these inputs to generate the final code that represents the Input      
            Image.
Output Code: This is the final product of the system, the actual code that has been generated to represent the input image. This code is intended to recreate the GUI from the input image in a format that can be rendered or executed, such as HTML/CSS for web interfaces.

DATA FLOW DIAGRAM:







##CLASS DIAGRAM:

ImageToCodeSystem 
- yoloModel: YOLOModel
- supervisedLearning: SupervisedLearning
- codeGenerator: CodeGenerator 
+ generateCodeFromImage(image: Image): Code 

↓uses
 YOLOModel 
- parameters: YOLOParameters
+ trainModel(dataset: LabeledDataset): void
+ detectObjects(image: Image): List<DetectedObject> 

↓
SupervisedLearning 
- model: TrainedModel 
+ trainModel(dataset: LabeledDataset): void 
+ predict(image: Image): PredictionResult

↓
 CodeGenerator 
+ generateCode(objects: List<DetectedObject>): Code

↓
YOLOParameters
- configuration: String
+ setConfiguration(config: String): void 
+ getConfiguration(): String

↓
 Image  
- pixels: Array[] 
+ load(): void      
+ getPixels(): Array[]

↓
LabeledDataset 
- images: List<Image>  
- labels: List<Label>
+ addExample(image: Image, label: Label): void

↓
DetectedObject 
- classLabel: String  
- boundingBox: BoundingBox
+ getClassLabel(): String 
+ getBoundingBox(): BoundingBox

↓
BoundingBox
- x: double
- y: double
- width: double
- height: double
+ getX(): double 
+ getY(): double
+ getWidth(): double
+ getHeight(): double 

↓
   Code 
- codeString: String
+ getCodeString(): String

SYSTEM TESTING

The purpose of testing is to discover errors. Testing is the process of trying to discover every conceivable fault or weakness in a work product. It provides a way to check the functionality of components, sub assemblies, assemblies and /or a finished product. It is the process of exercising software with the intent of ensuring that the software system meets its requirements and user expectations and does not fail in an unacceptable manner. There are various types of testing. Each test type addresses a specific testing requirement.

TYPES OF TESTS

Unit Testing:

Unit testing involves the design of test cases that validates that the internal program logic is functioning properly, and that program inputs produce valid outputs. All decision branches and internal code flow should be validated. It is the testing of individual software units of the application. It is done after the completion of an individual unit before integration. This is a structural testing that relies on knowledge of its construction and is invasive. Unit tests perform basic tests at component level and test a specific business process, application, and /or system configuration. Unit tests ensure that each unique path of a business process performs accurately to the documented specifications and contains clearly defined inputs and expected results.

Integration Testing:

Integration tests are designed to test integrated software components to determine if they actually run as one program. Testing is event driven and is more concerned with the basic outcome of screens or feels. Integration tests demonstrate that although the components were individually satisfactory,as shown by successful unit testing, the combination of components is correct and consistent. Integration testing is specifically aimed at exposing the problems that arise from the combination of components.

Functional test:

Functional tests provide systematic demonstrations that functions tested are available as specified by the business and technical requirements, system documentation,and user manual.
Functional testing is centered on the following items:
Valid Input : identified classes of valid input must be accepted. 
Invalid Input : identified classes of invalid input must be rejected.
Functions : identified functions must be exercised.
Output : identified classes of application outputs must be exercised.
Systems/Procedures : 

Interfacing systems or procedures must be invoked.

Organization and preparation of functional tests is focused on requirements,  key functions, or special test cases. In addition, systematic coverage pertaining to identifying business process flows; data fields , predefined processes and successive processes must be considered for testing. Before functional testing is complete, additional tests are identified and the effective value of current tests is determined.   
                                                 
System Testing:

System Testing ensures that the entire integrated software system meets requirements; it tests a configuration to ensure known and predictable results. An example of system testing is the configuration oriented system integration test. System testing is based on process descriptions and flows, emphasizing pre-driven process links and integration  and points.
 
White Box Testing:

White box testing is a testing in which the software tester has knowledge of the inner workings , structure and language of the software , or at least its  purpose purposefully it is used to test areas that cannot be reached from a blackbox level.
 
Black Box Testing:

Black Box Testing is testing the software without any knowledge of the inner workings, structure or language of the module being tested. Black box test, as most other kinds of test, must be written from a definitive source document, such as specification or requirements document. It is a test in which the  software under test is treated as a black box.As you can “see” into it. The test provides inputs and responses to outputs without considering how the software works.

Unit Testing:

Unit testing is usually conducted as a part of the combined code and unit test phase of the software lifecycle, although it is not uncommon for coding and unit testing to be conducted as two distinct phases.
                                                
TEST STRATEGY AND APPROACH:

Field testing will be performed manually and functional testers will be written in  detail.
Test objectives
All field entries must work properly.
Pages must be activated from the identified link.
The entry screen, messages and responses must not be delayed.

Features to be tested 
Verify that the entries are of the correct format.
No duplicate entries should be allowed.
All links should take the user to the correct page.

Integration Testing:

Software integration testing is the incremental integration testing of two or more integrated software components on a single platform to produce failures caused by interface defects.
The task of the integration test is to check that components or software applications, eg: components in a software system or -one step up- software applications at the company level - interact without error.
Test Results: all the test cases mention about passing successfully.no defects encountered.

Acceptance Testing:

User acceptance testing is a critical phase of any project and requires significant participation by the end user. It also ensures that the system needs the functional
Requirements.
Test Results:all the test cases mentioned above pass successfully.no defects encountered. 














##CODE

Module 1: Custom Training the YOLO model 

Yolov8_v2_web_ui_analyst_code_generate.ipynb

# Pip install method (recommended)
'''bash
!pip install ultralytics==8.0.20

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

%cd /content/drive/MyDrive/v2_web_ui_analyst_code_generate

!yolo task=detect mode=train model=yolov8m.pt data= data.yaml epochs=50 imgsz=640 plots=True

#validating the model
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml


#inference with custom model
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=data/test/images

import glob
from IPython.display import Image, display

for image_path in glob.glob('runs/detect/predict/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
 '''     

Output:
Validate Custom Model:


Fig 13: validation of custom model
Inference With Custom Model:


Fig 14: Inference the details of custom model

Batch Generated:


Fig 15: The output representing the object detection in specific batch

Code_pred.py:
'''bash
import json
import torch
from ultralytics import YOLO

# Path to the custom trained YOLOv8 model weights
weights_path = './runs/detect/train/weights/best.pt'

# Load the model
model = YOLO(weights_path)

# Image path for inference
image_path = 'ss_jatin.png'

# Perform inference
results = model.predict(image_path, conf=0.4, imgsz=(640, 640))

# Extract the coordinates and class names of detected objects
objects = []
unique_classes = set()
for result in results:
    boxes = result.boxes  # Boxes object for box outputs

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        x_min, y_min, x_max, y_max = box
        class_name = result.names[int(class_id)]
        unique_classes.add(class_name)
        object = {
            "x": (x_min + x_max) / 2,
            "y": (y_min + y_max) / 2,
            "x_min": x_min,
            "y-min": y_min,
            "x_max": x_max,
            "y-max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "class": class_name
        }
        objects.append(object)

# Write html file
f = open('new_test.html', 'w') 

html_start ="""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        .detected-object {
            position: absolute;
            border: 1px solid red; 
        }
        
        """

f.write(html_start) 
f.close() 
f = open('new_test.html', 'a') 


max_buttons=(str(b+1) for b in range(len(objects)))
for ob in objects:
    if ob['class'] != 'button':
        txt= f"#{ob['class']}"+ "{"+f"""
            left: {int(ob['x_min'].item())}px; 
            top: {int(ob['y-min'].item())}px; 
            width: {int(ob['width'].item())}px; 
            height: {int(ob['height'].item())}px;          
         """+"""}
         
         """
        f.write(txt)
    else:
        txt= f"#{ob['class']}"+next(max_buttons)+ "{"+f"""
            left: {int(ob['x_min'].item())}px; 
            top: {int(ob['y-min'].item())}px; 
            width: {int(ob['width'].item())}px; 
            height: {int(ob['height'].item())}px;          
         """+"""}
         
         """
        f.write(txt)        
            
            
html_head2body=""" 
    </style>
</head>
<body>
        """
f.write(html_head2body) 

max_buttons=(str(b+1) for b in range(len(objects)))
for ob in objects:
    if ob['class'] != 'button':
        txt= f"""<div id="{ob['class']}" class="detected-object"></div>
        """
        f.write(txt)
    else:
        txt= f"""<div id="{ob['class']}{next(max_buttons)}" class="detected-object"></div>
        """
        f.write(txt)   
        

html_end="""
</body>

</html>
"""
f.write(html_end) 
f.close() 
'''
App.py:
'''bash
from flask import Flask, render_template, request, send_file, url_for
import os
import torch
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/'

# Load YOLO model (adjust the path as needed)
model = YOLO('best_v3.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    confidence = float(request.form.get('confidence', 0.5))
    overlap = float(request.form.get('overlap', 0.3))
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_image_filename, html_filename = process_image(file_path, confidence, overlap)
        return render_template('download.html', 
                               image_filename=processed_image_filename, 
                               html_filename=html_filename)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_image_filename, html_content = process_image(file_path, confidence, overlap)
        return render_template('download.html', 
                               image_filename=processed_image_filename, 
                               html_content=html_content)

def draw_box(image, box, label):
    # Draw a rectangle and label on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
    draw.text((box[0] + 10, box[1] + 10), label, fill="red")

def process_image(image_path, confidence, overlap):
    # Load image with PIL
    img = Image.open(image_path)

    # Perform inference
    results = model.predict(img, conf=confidence, iou=overlap, imgsz=(640, 640))

    # Initialize an array to store details of detected objects
    objects = []

    # Draw boxes and labels on the image and store details
    for result in results:
        boxes = result.boxes
        for box, class_id in zip(boxes.xyxy, boxes.cls):
            # Draw each box on the image
            draw_box(img, box, result.names[int(class_id)])
            # Store object details
            x_min, y_min, x_max, y_max = box
            class_name = result.names[int(class_id)]
            objects.append({
                "x_min": x_min.item(),
                "y_min": y_min.item(),
                "width": (x_max - x_min).item(),
                "height": (y_max - y_min).item(),
                "class": class_name
            })


    # Save the processed image
    processed_image_filename = 'processed_' + os.path.basename(image_path)
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_filename)
    img.save(processed_image_path)
    
    # Generate and save HTML file with detected objects
    html_filename = 'new_test_' + os.path.splitext(os.path.basename(image_path))[0] + '.html'
    html_file_path = os.path.join(app.config['PROCESSED_FOLDER'], html_filename)
    with open(html_file_path, 'w') as f:
    # Write the start of the HTML and CSS
                f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Document</title>
                <style>

                    .detected-object {
                        position: absolute;
                        border: 1px solid red;
                    }
                """)
                
                # Write CSS for each object
                for i, ob in enumerate(objects):
                    f.write(f"""
                    #{ob['class']}{i} {{
                        left: {int(ob['x_min'])}px;
                        top: {int(ob['y_min'])}px;
                        width: {int(ob['width'])}px;
                        height: {int(ob['height'])}px;
                    }}
                    """)
                
                # End of CSS and start of body
                f.write("""
                </style>
            </head>
            <body>
                """)

                # Write HTML elements for each object
                for i, ob in enumerate(objects):
                    if ob['class'] == 'image':
                        f.write(f'<img id="{ob["class"]}{i}" class="detected-object" src="default.png" alt="Default Image">\n')
                    elif ob['class'] == 'text':
                        f.write(f'<div id="{ob["class"]}{i}" class="detected-object">Some random text</div>\n')
                    elif ob['class'] == 'button':
                        f.write(f'<button id="{ob["class"]}{i}" class="detected-object">Button</button>\n')
                    elif ob['class'] == 'header':
                        # style="background-color: #2196F3; padding: 15px 20px; overflow: hidden;">
                        f.write(f"""<div id="{ob["class"]}{i}" style = "background-color: #2196F3;" class="detected-object">
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">Home</a>
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">About</a>
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">Contact</a>
                                </div>
                            </header>\n""")
                    elif ob['class'] == 'footer':
                        f.write(f'<footer id="{ob["class"]}{i}" class="detected-object">Footer content</footer>\n')
                    elif ob['class'] == 'card':
                        f.write(f'<div id="{ob["class"]}{i}" class="detected-object"><img id="{ob["class"]}{i}" src="default.png" alt="Default Image"><h3 class="detected-object">Text for card</h3></div>\n')
                    elif ob['class'] == 'search_bar':
                        f.write(f"""
                        <div id="{ob["class"]}{i}" class="detected-object" >
                            <input id="{ob["class"]}{i} type="text" placeholder="Search">
                            <buttoni d="{ob["class"]}{i} >Search</button>
                        </div>
                        """)
                
                # End of HTML
                f.write("""
            </body>
            </html>
            """)

     
   
    return processed_image_filename, html_filename



@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(os.getcwd(), app.config['PROCESSED_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
'''
Module 3: Front-end UI code 

Index.html:
'''bash
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Upload Image</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="shortcut icon" href="/static/layers.png" type="image/x-icon">
    <style>
        body {
            
            background-color: #081c10;
            opacity: 0.75;
        }

        .navbar {
            padding: 14px 16px;
            background-color: #081c10;
            border-bottom: 1px solid #ADBDAB;
        }

        .navbar a {
            color: #ADBDAB;
            font-size: 1.5rem;
            font-weight: bold;
            text-decoration: none;
            margin-right: 20px;
            transition: color 0.3s ease-in-out, transform 0.3s ease-in-out;
        }

        .navbar a:hover {
            color: #ffffff;
            transform: scale(0.95);
        }

        .container {
            padding: 14px 16px;
        }

        .form-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 50px;
        }

        .preview-container {
            margin-top: 20px;
        }

        #image-preview {
            max-width: 100%;
            max-height: 300px;
        }

        .hidden {
            display: none;
        }

        footer {
            background-color: #081c10;
            color: white;
            opacity: 0.5;
            text-align: center;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
    </style>
</head>
<body>

    <nav class=" py-14 px-16">
        <div class="container mx-auto flex justify-between items-center ">
            <!-- Logo or Website Name -->
            <a href="/home"
                class="text-gray-300 text-3xl  font-semibold hover:text-[#ADBDAB] hover:scale-95">Image2code</a>

            <!-- Navigation Links -->
            <div class="flex space-x-4 px-4">
                <a href="/home" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">Home</a>
                <a href="/" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">Image2code</a>

                <a href="/aboutus" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">About Us</a>
                <!-- Add more links as needed -->
            </div>
        </div>
        <br>
        <hr>
    </nav>
<div class="flex flex-col md:flex-row">
<div class="container">
    <h1 class="text-3xl font-bold py-4 px-4 text-white">Upload Image</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" class="py-4 font-semibold text-white" onchange="handleFileSelect(this)"><br><br>
        <label for="confidence" class="font-semibold text-white">Confidence:</label>
        <input class="py-2 px-3 w-1/2 border border-[#081c10] rounded-lg shadow-sm text-center" type="number"
               name="confidence" step="0.01" min="0" max="1" value="0.5"><br><br>
        <label for="overlap" class="font-semibold text-white">IOU Overlap:</label>
        <input class="py-2 px-3 w-1/2 border border-[#081c10] rounded-lg shadow-sm text-center" type="number"
               name="overlap" step="0.01" min="0" max="1" value="0.3"><br><br>
        <input class="py-2 bg-[#dde0df] rounded-full space-between tracking-tight px-8 hover:opacity-75"
               type="submit" value="Upload">
    </form>
</div>
    <div class="preview-container pr-28">
        <img id="image-preview" src="#" alt="Image Preview" class="hidden">
    </div>

</div>

<footer>
    <div class="container mx-auto text-center">
        <p class="mb-2">Syed Nazeeruddin Yousuf | Mohammed Abdul Razzack | Ragella Manish | Sai Kiran</p>
        <p>&copy; 2023 Your Website. All rights reserved.</p>
    </div>
</footer>

<script>
    function handleFileSelect(input) {
        const preview = document.getElementById('image-preview');
        const file = input.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.classList.remove('hidden');
            };

            reader.readAsDataURL(file);
        } else {
            preview.src = '#';
            preview.classList.add('hidden');
        }
    }
</script>

</body>
</html>
'''
Download.html:
'''bash
<!DOCTYPE html>
<html>
<head>
    <title>Download File</title>
    <script src="https://cdn.tailwindcss.com"></script>

</head>
<body class="bg-[#081c10] opacity-75 text-white"> 
    <nav class=" py-14 px-16">
        <div class="container mx-auto flex justify-between items-center ">
            <!-- Logo or Website Name -->
            <a href="/home"
                class="text-gray-300 text-3xl  font-semibold hover:text-[#ADBDAB] hover:scale-95">Image2code</a>

            <!-- Navigation Links -->
            <div class="flex space-x-4 px-4">
                <a href="/home" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">Home</a>
                <a href="/" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">Image2code</a>

                <a href="/aboutus" class="text-gray-300 text-xl font-semibold hover:text-[#ADBDAB]">About Us</a>
                <!-- Add more links as needed -->
            </div>
        </div>
        <br>
        <hr>
    </nav>
    <div class="container py-2 px-16">
        <div>
    <h1 class="text-3xl font-bold py-4 px-4">Download Processed File</h1>
    <img style="height: 500px;width: 700px" src="{{ url_for('static', filename=image_filename) }}" alt="Predicted Image"><br><br>
    <a href="/download/{{ html_filename }}" class="py-2 bg-[#dde0df] rounded-full space-between tracking-tight px-8 hover:opacity-50 text-black">Download HTML File</a></div>
</html>
'''














##CONCLUSION

In this, we introduced Image2Code, a unique approach that uses a single GUI picture as input to produce computer code. Although our study shows how such a system may be used to automate the process of putting GUIs into practice, we have just begun to explore the possibilities. Our model was trained on a tiny dataset and has a comparatively modest number of parameters. Training a larger model on a larger amount of data , using more number of classes and over a longer number of epochs might result in a considerable improvement in the quality of the resulting code.Since simple straightforward approach just gives simple representation of the website, it is unable to provide any meaningful information about the connections between the elements etc. In order to reduce error in the resulting code, it would be possible to infer the connections between elements and implement an OCR model to extract the relevant text for the input image for more accurate generation of the code. Furthermore, the number of classes that the model can predict is limited since the size of the dataset is small and does not scale easily  to very large dataset. Since the size of the dataset is small and also the number of classes taken are limited in number , the project can provide basic structure code for the images provided but not for complex ones.

In future work,
Many more classes and other approaches like reinforcement learning can be used to achieve this goal in a more efficient and better manner to get much better results , but it also results in complicating the structure and having a much deeper understanding of how these models and technologies work together.


















REFERENCES

[1] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.
 
[2] M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow. Deepcoder: Learning to write programs. arXiv preprint arXiv:1611.01989, 2016.
 
[3] B. Dai, D. Lin, R. Urtasun, and S. Fidler. Towards diverse and natural image descriptions via a conditional gan. arXiv preprint arXiv:1703.06029, 2017.
 
[4] J. Donahue, L. Anne Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell. Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2625–2634, 2015.
 
[5] A. L. Gaunt, M. Brockschmidt, R. Singh, N. Kushman, P. Kohli, J. Taylor, and D. Tarlow. Terpret: A probabilistic programming language for program induction. arXiv preprint arXiv:1608.04428, 2016. 

[6] F. A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: Continual prediction with lstm. Neural computation, 12(10):2451–2471, 2000. 

[7] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680, 2014. 

[8] A. Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013. 

[9] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735– 1780, 1997.
 
[10] A. Karpathy and L. Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3128–3137, 2015.
 
[11] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.
 
[12] W. Ling, E. Grefenstette, K. M. Hermann, T. Kocisk ˇ y, A. Senior, F. Wang, and P. Blunsom. ` Latent predictor networks for code generation. arXiv preprint arXiv:1603.06744, 2016.

[13] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013.
 
[14] T. A. Nguyen and C. Csallner. Reverse engineering mobile application user interfaces with remaui (t). In Automated Software Engineering (ASE), 2015 30th IEEE/ACM International Conference on, pages 248–259. IEEE, 2015.
 
[15] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. In Proceedings of The 33rd International Conference on Machine Learning, volume 3, 2016.
 
[16] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. arXiv preprint arXiv:1312.6229, 2013.
 
[17] R. Shetty, M. Rohrbach, L. A. Hendricks, M. Fritz, and B. Schiele. Speaking the same language: Matching machine to human captions by adversarial training. arXiv preprint arXiv:1703.10476, 2017.
 
[18] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
 
[19] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.
 
[20] T. Tieleman and G. Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 2012.
 
[21] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3156–3164, 2015.
 
[22] K. Xu, J. Ba, R. Kiros, K. Cho, A. C. Courville, R. Salakhutdinov, R. S. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, volume 14, pages 77–81, 2015.
 
[23] L. Yu, W. Zhang, J. Wang, and Y. Yu. Seqgan: sequence generative adversarial nets with policy gradient. arXiv preprint arXiv:1609.05473, 2016.
 
[24] W. Zaremba, I. Sutskever, and O. Vinyals. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329, 2014.
 
[25] H. Zhang, T. Xu, H. Li, S. Zhang, X. Huang, X. Wang, and D. Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. arXiv preprint arXiv:1612.03242, 2016.

[26] Taneem, Jude, Dr.Zakira Inayat. , sketch2code 2022
