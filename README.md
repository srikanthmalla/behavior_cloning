# BEHAVIOR Cloning
### Model Architecture and Training Strategy

| Layer (type) |  Output Shape  | Param  | 
| ------------- |:-------------:| -----:|
|lambda_1 (Lambda)  | (None, 160, 320, 3)    | 0 |        
|cropping2d_1 (Cropping2D)| (None, 65, 320, 3)| 0|         
|conv2d_1 (Conv2D) |    (None, 61, 316, 6) | 456|       
|max_pooling2d_1 (MaxPooling2D)| (None, 30, 158, 6)|  0 |        
|conv2d_2 (Conv2D)|  (None, 26, 154, 6)|    906|       
|max_pooling2d_2 (MaxPooling2D)| (None, 13, 77, 6)|  0|         
|flatten_1 (Flatten) |  (None, 6006) |    0  |       
|dense_1 (Dense) |        (None, 120)   |      720840    |
|dense_2 (Dense) |        (None, 84)    |      10164     |
|dense_3 (Dense) |        (None, 1)     |          85   |     

Final response:

![alt text](videos/final.gif)

For better HD video:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=rpTw07datrc
" target="_blank"><img src="http://img.youtube.com/vi/rpTw07datrc/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="720" height=AUTO border="10" /></a>
