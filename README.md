# blur-detection-alexnet

  This project was built wishing to detect if a text image is blur or not. It was built using Keras and Tensorflow as backend. And AlexNet as the network structure. Final accuracy on validation set reached 0.97104.  
  **2019.06.18 Update: used another test set with around 2100 pictures to run the evaluation. Received accuracy: 0.721 and recall: 0.974.**

****
*About python scripts.*  
  ***cnn.py***: use keras and tensorflow as backend combining a AlexNet's network structure. Only 1 gpu is used in this python script to run the training. If you want to use more than 1 gpu, please change the code with your needs. The weight is saved to the local directory with suffix '.h5'.  
    
  ***imgProcess.py***: this script is used to handle the input images' size as I orginally trained my network with size (512,512). Image is processed by being pasted to a newly created black image with a larger size(both width and height are multiples of 512) to form a new image. The new image is then cropped to multiple pieces of sub-images, all of which are of size (512,512). These sub-images are eventually saved to a newly created directory called slice.  
    
  ***noob_recog.py***: load the weight file and run the recognition to each of the sliced sub-pictures. Result will be saved in a txt file called _predict_prob.txt_ in the format of (image name + space + predict probability + space + true label). It seems a little bit messy here but you can see the picture above as reference.  
  <p align="center">
	<img src="https://github.com/SixTRaps/blur-detection-alexnet/blob/master/predict_result_template.png" alt="Sample"  width="508" height="692">
	<p align="center">
		<em>predict result template</em>
	</p>
</p>  

  ***concat.py***: concat the sub-pictures and calculate the average predict probability for each picture. Result will be saved in _average_accuracy.txt_.  
  
  ***eval.py***: read the contents in _average_accuracy.txt_ and calculate the accuracy and recall.
