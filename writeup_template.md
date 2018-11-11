# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[interst_mask]: ./test_images_output/interst_mask.jpg "interst_mask"
[uninterst_maxk]: ./test_images_output/not_interst_mask.jpg "uninterst_maxk"
[solidWhiteCurve]: ./test_images_output/solidWhiteCurve.jpg "solidWhiteCurve"
[solidWhiteRight]: ./test_images_output/solidWhiteRight.jpg "solidWhiteRight"
[solidYellowCurve]: ./test_images_output/solidYellowCurve.jpg "solidYellowCurve"
[solidYellowCurve2]: ./test_images_output/solidYellowCurve2.jpg "solidYellowCurve2"
[solidYellowLeft]: ./test_images_output/solidYellowLeft.jpg "solidYellowLeft"
[whiteCarLaneSwitch]: ./test_images_output/whiteCarLaneSwitch.jpg "whiteCarLaneSwitch"

---

### Reflection

### 1. 实现过程描述  
我将pipeline实现为`lane_line_pipeline`函数，其内部处理过程分为：

#### （1）灰度化     

是为了获取灰度图片，便于canny函数使用。

#### （2）高斯平滑  
增加一层平滑，调整canny检测出的边缘的粗细。

#### （3）canny边缘检测  
这里使用50和150作为阈值进行检测，该取值是多次试验得到的。

#### （4）设定区域  
为了防止因为视频或者图片的大小对标记曲线的影响，我将尺寸为（540,960）的画幅作为标准画幅，并贵其他尺寸的画幅进行缩放；
为了防止引擎盖上曲线对检测的影响，将遮罩区域底边向上移动50像素（在 540，960，图像中）；
为了防止道路两旁道牙对检测的影响，将遮罩区域向里收缩100像素（在 540，960，图像中）；

```
vertices = np.array([[(0+ int(100/540*imshape[0]),imshape[0]- int(50/540*imshape[0])),
	((imshape[1]/2 - int(10/540*imshape[0])), imshape[0]/2 + int(50/540*imshape[0])), 
	((imshape[1]/2 + int(10/540*imshape[0])), imshape[0]/2 + int(50/540*imshape[0])), 
	(imshape[1]- int(100/540*imshape[0]),imshape[0]- int(50/540*imshape[0]))]], dtype=np.int32)
```
![][interst_mask]

为了避免 Optional Challenge 中出现的，车道线之间以及车头前面的线条的影响，使用非感兴趣区域进行遮罩：
```
def region_of_not_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.ones_like(img)*255
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
    else:
        ignore_mask_color = 0
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # plt.imshow(mask)
    # plt.show()
    # mpimg.imsave('test_images_output/not_interst_mask.jpg',mask)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
![][uninterst_maxk]
#### （5）霍夫曼变换并绘制标记线
为了防止因为视频或者图片的大小对标记曲线的影响，我将`threshold`,`min_line_length`,`max_line_gap`三个参数做了缩放，缩放基准尺寸为（540,960）；  
为了使得车道线的标记直线尽可能地延长，我将`max_line_gap`的值设置的较大，以连接各段车道线；  
我将`min_line_length`和`threshold`设置的较大是为了排除短小的标记线的影响；  
```
    threshold = int(5/540*imshape[0])  
    min_line_length = int(120/540*imshape[0]) 
    max_line_gap = int(110/540*imshape[0]) 
```
为了使得车道线的标记直线得更加准确，并且标记线更加平滑，我先计算了`hough_lines`函数返回的各条直线的斜率，并将斜率介于0.5至0.99之间的直线视为右车道线，斜率介于-0.5至-0.99之间的直线视为左车道线；之后分别将左右车道线集合中的直线进行平均处理；  
为了使车道线能够延伸到图片边缘以及中心，采用fit函数拟合检测到的车道线，然后计算边缘及中心的左边点，最后将计算出的坐标点绘制出来。
```
def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    imshape = img.shape
    have_left_line = False
    left_x1 = []
    left_x2 = []
    left_y1 = []
    left_y2 = []
    
    have_right_line = False
    right_x1 = []
    right_x2 = []
    right_y1 = []
    right_y2 = []
    
    if (lines is not None) and len(lines) != 0:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if -0.99 < (y2-y1)/(x2-x1) < -0.5:
                    have_left_line = True
                    left_x1.append(x1)
                    left_x2.append(x2)
                    left_y1.append(y1)
                    left_y2.append(y2)
                elif 0.5 < (y2-y1)/(x2-x1) <0.99:
                    have_right_line = True
                    right_x1.append(x1)
                    right_x2.append(x2)
                    right_y1.append(y1)
                    right_y2.append(y2)
    
    
    if have_left_line:
        left_x1_avg = int(np.average(left_x1))
        left_x2_avg = int(np.average(left_x2))
        left_y1_avg = int(np.average(left_y1))
        left_y2_avg = int(np.average(left_y2))

        fit_left = np.polyfit([left_x2_avg,left_x1_avg],[left_y2_avg,left_y1_avg],1)

        bottom_left_x = int((imshape[0]- fit_left[1])/fit_left[0])
        bottom_left_y = int(imshape[0])
        
        top_left_x = int((imshape[0]/2 + 50/540*imshape[0]  - fit_left[1])/fit_left[0])
        top_left_y = int(imshape[0]/2 + 50/540*imshape[0])

        # print left line  
        cv2.line(img,(top_left_x, top_left_y ), (bottom_left_x, bottom_left_y), color, thickness)
        
        
    if have_right_line:
        right_x1_avg = int(np.average(right_x1))
        right_x2_avg = int(np.average(right_x2))
        right_y1_avg = int(np.average(right_y1))
        right_y2_avg = int(np.average(right_y2))
        
        fit_right = np.polyfit([right_x1_avg, right_x2_avg],[right_y1_avg, right_y2_avg],1)

        bottom_right_x = int((imshape[0] - fit_right[1])/fit_right[0])
        bottom_right_y = int(imshape[0])

        top_right_x = int((imshape[0]/2 + 50/540*imshape[0] - fit_right[1])/fit_right[0])
        top_right_y = int(imshape[0]/2 + 50/540*imshape[0])

        # print right line
        cv2.line(img,(top_right_x, top_right_y), (bottom_right_x, bottom_right_y) , color, thickness)    
```

#### （6）将标记图片和原始图片合并


#### 静态图片标记结果
![][solidWhiteCurve]

![][solidWhiteRight]

![][solidYellowCurve]

![][solidYellowCurve2]

![][solidYellowLeft]

![][whiteCarLaneSwitch]

### 2. 不足之处

- 存在一定的概率无法识别到车道线；
- 标记线为直线，无法标记弯道的曲线车道线。


### 3. Suggest possible improvements to your pipeline

- 一种改进措施是，适当降低hough_liine函数中对检测车道线标记线最小长度的要求，将多个较短的标记线，更具斜率的相似性合并为一条较长的直线；
- 另一种改进措施是，不使用执行进行标记，对检测到的边缘点进行曲线拟合；
