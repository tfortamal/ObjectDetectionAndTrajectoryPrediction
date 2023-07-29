## ObjectDetectionAndTrajectoryPrediction
# Object Detection and Trajectory prediction using Faster RCNN and Kalman filter
# 1. **Abstract**

This project focuses on object tracking and trajectory prediction using Faster R-CNN and the Kalman filter. Object detection is a vital step in object tracking, where region proposal algorithms are commonly used in state-of-the-art object detection networks. However, these region proposal computations can be computationally expensive. The motivation behind this project is to develop a robust and efficient object-tracking system with accurate trajectory prediction. By integrating Faster R-CNN, which excels in object detection, and the Kalman filter, a powerful prediction algorithm, we aim to enhance tracking accuracy and efficiency.

The methodology involves employing Faster R-CNN for object detection using a pre-trained model fine-tuned on a specific dataset. Subsequently, the Kalman filter is implemented to predict object trajectories based on detected states, leading to smoother and more reliable tracking results. The testing phase shows impressive object detection results with the pre-trained model on the COCO dataset, while implementing the Kalman filter further enhances object tracking performance. Future development involves using a larger dataset to improve object detection and exploring advanced algorithms for more versatile real-world applications.

# 2. I**ntroduction**

Object tracking is a fundamental problem in computer vision, with wide-ranging applications in various fields, including surveillance, robotics, autonomous vehicles, augmented reality, and human-computer interaction. The ability to accurately track and predict the trajectory of objects in real time has become increasingly essential as these technologies continue to advance.

The primary motive of this project is to develop an efficient and robust object tracking and trajectory prediction system using state-of-the-art techniques. Object tracking involves continuously locating and following objects as they move through a video sequence or a sequence of images. While object detection can be accomplished in individual frames, accurate tracking requires maintaining identity across consecutive frames and predicting future object locations.

The choice of using Faster R-CNN for object detection and the Kalman filter for trajectory prediction arises from their respective strengths and complementary functionalities. Faster R-CNN, an extension of the R-CNN and Fast R-CNN methods, revolutionized object detection by introducing a Region Proposal Network (RPN) that efficiently generates region proposals for potential objects. The RPN shares convolutional features with the object detection network, enabling end-to-end training and significantly reducing computation time.

In contrast, the Kalman filter is a recursive and optimal algorithm used to estimate the state of dynamic systems from a series of noisy measurements. It is particularly well-suited for tracking applications due to its ability to predict the future state of an object based on past observations and motion models. By fusing the Kalman filter's predictions with the object detections from Faster R-CNN, we aim to achieve accurate and smooth object tracking over time.

The integration of Faster R-CNN and the Kalman filter holds the promise of overcoming some of the limitations faced by traditional tracking algorithms. Faster R-CNN provides highly accurate and precise object detection, while the Kalman filter helps maintain continuity in tracking, even when detection results are occasionally missing or inaccurate.

The methodology involves first utilising Faster R-CNN for object detection. We employ a pre-trained model, which has been trained on the COCO dataset.

Once objects are detected in consecutive frames, the Kalman filter is implemented to predict their future trajectories. The filter takes the initial state of each tracked object from the detection results and uses motion models to estimate the object's positions in subsequent frames. The Kalman filter's strength lies in its ability to handle noisy measurements and smoothly predict object trajectories even in situations with partial or unreliable detections.

In the results section, we evaluate the performance of our system using the pre-trained Faster R-CNN model. The pre-trained model displays impressive object detection results on standard datasets, demonstrating its effectiveness in detecting objects in diverse scenarios. 

By integrating the Kalman filter into our object-tracking system, we aim to enhance tracking accuracy and robustness. The filter's predictions help bridge the gap between consecutive frames and ensure smoother trajectories, reducing jitter and increasing overall tracking stability.

In conclusion, this project endeavours to advance the field of object tracking and trajectory prediction by combining the power of Faster R-CNN and the Kalman filter. The synergy between these two techniques offers a promising approach to achieving accurate, real-time object tracking in diverse real-world scenarios. The knowledge gained from this project paves the way for future development, including the exploration of larger datasets and the integration of more sophisticated tracking algorithms to further improve object-tracking performance in various applications.


# 3. **Literature survey**

In this project, we conducted an extensive literature survey to gather insights into object detection and object tracking techniques. To gain a comprehensive understanding of the subject, we explored various research papers, articles, and educational resources, including YouTube videos.

### Object Detection and Object Tracking

We delved into several articles and research papers that discussed object detection and tracking algorithms extensively. The literature covered classical approaches as well as state-of-the-art deep learning-based methods. Understanding these techniques helped us grasp the evolution of object detection and tracking over the years.

### Links to Articles and Research Papers:

1. [A Brief Overview of R-CNN, Fast R-CNN, and Faster R-CNN](https://medium.com/mlearning-ai/a-brief-overview-of-r-cnn-fast-r-cnn-and-faster-r-cnn-9c6843c9ffc0)
2. [R-CNN, Fast R-CNN, Faster R-CNN, Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)
3. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
4. [DIVOTrack: A Novel Dataset and Baseline Method for Cross-View Multi-Object Tracking in DIVerse Open Scenes](https://arxiv.org/abs/2302.07676)
5. [Faster R-CNN Explained for Object Detection Tasks](https://blog.paperspace.com/faster-r-cnn-explained-object-detection/)

### Kalman Filter for Object Tracking

We also explored the Kalman filter, a powerful mathematical tool widely used for object tracking in computer vision applications. Understanding the principles and intuition behind the Kalman filter helped us integrate it effectively with the Faster R-CNN model to improve trajectory prediction and object tracking accuracy.

### Links to Resources:

1. [An Intuition about Kalman Filter for Computer Vision](https://www.analyticsvidhya.com/blog/2021/10/an-intuition-about-kalman-filter/)

By synthesising information from these diverse sources, we gained valuable insights into the theory and practical implementation of object detection and tracking algorithms. This knowledge guided us in designing and developing a robust system that combines the Faster R-CNN model with the Kalman filter to achieve accurate and efficient object tracking and trajectory prediction.


## 4.1 R-CNN, Fast R-CNN, and Faster R-CNN

In this section, we compare the key features and performance characteristics of R-CNN, Fast R-CNN, and Faster R-CNN:

| Model | Region Proposal | Speed (inference time) | End-to-End Training | Training Time | Object Detection Accuracy |
| --- | --- | --- | --- | --- | --- |
| R-CNN | External (Selective Search) | Slow | No | High | Moderate |
| Fast R-CNN | Internal (RoI Pooling) | Faster | Yes | Moderate | Moderate |
| Faster R-CNN | Internal (Region Proposal Network) | Fastest | Yes | Low | High |

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46265797-20df-43d6-a835-2abf4d8506b6/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dacdb630-66ea-4861-9868-5fab681e40cc/Untitled.png)

- R-CNN: The original R-CNN used an external region proposal algorithm (Selective Search) to generate potential object regions, resulting in slow inference time. It involved separately processing each region, making it computationally expensive. Although it achieved moderate object detection accuracy, it lacked end-to-end training, which affected its overall performance.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf78f61a-58c2-4d9b-ae40-6d24c72ea90d/Untitled.png)
    
- Fast R-CNN: Fast R-CNN addressed the speed and training issues of R-CNN by introducing RoI (Region of Interest) Pooling, allowing for internal region proposal and shared feature computation. This led to faster inference times and enabled end-to-end training, improving both training time and object detection accuracy.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59fa783f-8da2-41df-bd0c-2463fc3129f0/Untitled.png)
    
- Faster R-CNN: Building upon Fast R-CNN, Faster R-CNN further enhanced speed and accuracy by introducing the Region Proposal Network (RPN). The RPN shares convolutional features with the object detection network, enabling the model to generate region proposals internally and significantly reducing inference time. Faster R-CNN achieved the highest object detection accuracy among the three models, making it the preferred choice for real-time applications.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a23106d-d0e1-4d55-b618-dad46a14e27f/Untitled.png)
    

## 4.2 Object Detection using Faster R-CNN

Faster R-CNN is a two-stage object detection model that combines the advantages of both Fast R-CNN and the Region Proposal Network (RPN). Let's delve into the details of Faster R-CNN:

### Region Proposal Network (RPN):

The RPN is a small, fully convolutional network that is responsible for generating region proposals (candidate bounding boxes) for potential objects. The RPN slides a small window (typically 3x3) over the convolutional feature map and predicts whether an object is present inside that window or not. Additionally, it also regresses the coordinates of the bounding boxes corresponding to potential objects.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4e7bfa99-85f9-4e0e-a5d4-88c99b373a6a/Untitled.png)

The RPN uses anchor boxes of various sizes and aspect ratios to propose regions of interest. These anchor boxes act as predefined templates and assist in detecting objects of different scales and shapes.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff4a1bd5-9f04-4a56-8e32-fcd0d03e32ca/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3dc01a6-a85e-43a0-b50f-6d90741d9f21/Untitled.webp)

### Fast R-CNN Network:

After the RPN generates region proposals, the Faster R-CNN model uses these proposals to extract RoI features from the convolutional feature map. These RoI features are then fed into the Fast R-CNN network, which performs object classification and bounding box regression.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e3b4195-b284-49f9-9118-0a8b2ac7132c/Untitled.png)

The Fast R-CNN network has fully connected layers that can handle varying sizes of RoI features, allowing for object detection irrespective of the region's spatial dimensions. The model classifies objects into different categories and refines the bounding boxes to more accurately fit the detected objects.

Faster R-CNN allows end-to-end training of the entire system, optimising both the RPN and Fast R-CNN jointly. This integration enables faster and more accurate object detection compared to its predecessors.

## 4.3 Kalman Filter Implementation

## **What is Kalman Filter?**

The Kalman filter, a fundamental tool in statistics and control theory, is an algorithm designed to estimate the state of a system by incorporating a sequence of measurements taken over time, accounting for statistical noise. It achieves this by combining the predicted state of the system and the latest measurement in a weighted average. These weights are assigned based on the relative uncertainty of the values, with more confidence placed in measurements with lower estimated uncertainty. This process is also referred to as linear quadratic estimation.

The Kalman filter is an iterative algorithm used to estimate the state of a dynamic system from a series of noisy measurements. In the context of object tracking, the Kalman filter is employed to predict the future state of a tracked object based on its previous state and the observed measurements.

At each time step, the Kalman filter performs two main steps: the prediction step and the update step. In the prediction step, the filter estimates the future state of the object based on its current state and the system's motion model. It also predicts the covariance of the estimation error to account for uncertainties in the object's motion.

In the update step, the Kalman filter takes into account the measurements obtained from the object's detections. It updates the predicted state using the actual measurements, adjusting the estimated state based on the reliability of the measurements and the Kalman gain.

The Kalman filter dynamically adjusts its estimation based on the accuracy of the measurements and the motion model, making it robust in handling noisy or missing detections and ensuring smooth and accurate object tracking.

### Mathematical Explanation:

The Kalman filter can be represented by two main equations:

1. **Prediction Step:**
    - Predicted state estimate: Predict the next state of the system based on its current state and a motion model that describes how the system evolves over time. Use the state transition matrix (F) to project the current state into the future and account for the system's dynamics. If applicable, include a control input (u) to model any external influences on the system
    - Predicted error covariance: Update the state covariance matrix (P) to incorporate the uncertainty in the prediction. Use the process noise covariance matrix (Q) to account for uncertainty or noise in the system's dynamics.
2. **Update Step:**
    - Kalman gain: Calculate the Kalman Gain (K), which determines how much weight to give to the prediction and the measurements during the update step. The Kalman Gain is a crucial factor that balances the confidence in the prediction and the measurements.
    - Residual (measurement error): Calculate the measurement residual (y) by finding the difference between the actual measurements and the predicted measurements (based on the prediction).
    - Updated state estimate: Update the state estimate (x) using the Kalman Gain, the measurement residual, and the predicted state. This step combines the prediction and the measurements to produce a more accurate and refined state estimate
    - Updated error covariance: Update the state covariance matrix (P) to reflect the reduced uncertainty after incorporating the measurements
    

The Kalman gain determines the weighting of the predicted state and the measured state during the update step. It balances the influence of the motion model and the actual measurements, allowing the Kalman filter to adapt to changing conditions and handle uncertainties effectively.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/de40cd0a-702e-4eac-9f7c-4b8d02396e0e/Untitled.png)

By iteratively applying these prediction and update steps, the Kalman filter refines the object's state estimate over time, resulting in improved object tracking and trajectory prediction.

# 5. M**ethodology**

## ****5.1 Object Detection using Faster R-CNN****

To achieve precise and efficient object detection, we employ the state-of-the-art Faster R-CNN model. Let's walk through the step-by-step process of how objects are detected using the Faster R-CNN model, supported by images for better understanding.

### **Step 1: Input Image**

The process begins with providing an input image to the Faster R-CNN model. The model takes this image as input and processes it to identify objects present in the scene.

### **Step 2: Backbone Convolutional Network**

The input image is passed through the backbone convolutional network, which extracts hierarchical features from the image. The backbone network, often based on architectures like ResNet or VGG16, is pre-trained on large-scale datasets like ImageNet to capture general image features.

### **Step 3: Region Proposal Network (RPN)**

The feature map generated by the backbone network is then used by the Region Proposal Network (RPN). The RPN is responsible for proposing potential object regions in the image. It slides a small window (typically 3x3) over the feature map, generating region proposals and predicting whether each region contains an object or not.

### **Step 4: Anchor Boxes**

To propose regions of interest, the RPN utilises anchor boxes of various sizes and aspect ratios. These anchor boxes act as predefined templates and help in detecting objects of different scales and shapes. The RPN generates multiple region proposals, each associated with an anchor box.

### **Step 5: Region of Interest (RoI) Pooling**

The region proposals generated by the RPN are then passed to the RoI pooling layer. The RoI pooling layer aligns the proposals with the underlying feature map, ensuring a fixed-size feature representation for each region. This step allows the model to handle regions of different spatial dimensions and prepares the proposals for further processing in the Fast R-CNN network.

### **Step 6: Fast R-CNN Network**

The RoI features are now fed into the Fast R-CNN network, which performs object classification and bounding box regression. The Fast R-CNN network, fully connected, efficiently takes the RoI features as input, enabling accurate classification and localisation of objects within the region proposals.

### **Step 7: Object Classification and Bounding Box Regression**

In the Fast R-CNN network, the model classifies the objects within each region proposal into different categories and refines the bounding boxes to precisely fit the detected objects. The bounding box regression ensures accurate localisation of the objects, resulting in more reliable object detection.

By integrating the RPN for region proposal generation and the Fast R-CNN network for object classification and bounding box regression, the Faster R-CNN model achieves fast and accurate object detection, making it a powerful choice for a wide range of computer vision tasks.

## **5.2 Object Trajectory Prediction using Kalman Filter**

Once objects are successfully detected in consecutive frames using Faster R-CNN, the trajectory prediction stage comes into play, leveraging the Kalman filter to enhance tracking accuracy. The Kalman filter is fed with the detected bounding box coordinates and centroid as essential features for predicting the future trajectories of the objects.

The Kalman filter is a powerful prediction algorithm that incorporates motion models and measurement updates to estimate the future states of tracked objects. The model efficiently handles uncertainties and noisy measurements by iteratively predicting and updating the object's state. By fusing the Kalman filter's predictions with the object detections from Faster R-CNN, we ensure seamless integration of object tracking and trajectory prediction.

The Faster R-CNN's bounding box coordinates and centroid serve as valuable inputs to the Kalman filter. They provide essential spatial information about the object's position and its centre, enabling the Kalman filter to make more accurate predictions about the object's future locations. This fusion of object detection and the Kalman filter's trajectory prediction enhances the overall tracking performance, ensuring smoother and more reliable tracking over time.

The combination of Faster R-CNN for object detection and the Kalman filter for trajectory prediction forms a cohesive methodology that excels in accurately tracking objects in dynamic environments. This robust framework opens up new possibilities for real-world applications, including surveillance, autonomous vehicles, and robotics, where accurate object tracking and trajectory prediction are crucial for decision-making processes.

# 6. R**esult**

In this section, we present the results obtained from our object detection and trajectory prediction system using the Faster R-CNN model integrated with the Kalman filter. The system was evaluated on various image and video datasets to assess its performance in different scenarios.

### **6.1 Object Detection in Images using Pre-trained Faster R-CNN**

We tested the Faster R-CNN model on static images containing multiple objects of different classes. The pre-trained Faster R-CNN model demonstrated impressive object detection capabilities, accurately identifying and localising objects within the images.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/72e65d22-9e95-481b-b6f3-4aa66bcd8934/Untitled.jpeg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5cefb8d7-8534-4dbd-9bc6-d70aded6ad09/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d20a9f4-308e-4fee-a45c-aa4354cb8b33/Untitled.jpeg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0c70da34-e852-4f23-8fe5-24966c40ba26/Untitled.png)

### **6.2 Implementing Pre-trained Faster R-CNN for Object Detection in Videos**

The Faster R-CNN model was further applied to video sequences to detect and track objects in dynamic scenes. The model efficiently detected objects across consecutive frames, showcasing its real-time object detection capabilities.

Here is a sample test result video:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ddddc019-0332-4c66-b2d7-af33a1e27e6c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a07d2b1b-d008-47ca-afe6-32a718fc0adc/Untitled.png)

**[Link to Test Result Video - Object Detection in Videos](https://link-to-video.mp4/)**

### **6.3 Equipping Faster R-CNN Model with Kalman Filter for Trajectory Prediction**

To enhance object tracking accuracy, we integrated the Faster R-CNN model with the Kalman filter for trajectory prediction. The Kalman filter successfully predicted the future positions of the detected objects, resulting in smoother and more consistent object tracking across frames.
# 7. C**onclusion**

In conclusion, this project demonstrates the successful integration of Faster R-CNN for object detection and the Kalman filter for trajectory prediction, resulting in an efficient and accurate object-tracking system. However, the limitations of our fine-tuned model on a small dataset suggest the need for a larger and more diverse dataset to further improve the object detection performance.

Future development involves acquiring a more extensive dataset and fine-tuning larger models to achieve state-of-the-art results in various scenarios. Additionally, exploring other advanced object detection and tracking algorithms can lead to even more robust and versatile systems for real-world applications.
