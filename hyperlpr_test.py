#导入包
from hyperlpr import *
#导入OpenCV库
import cv2


# 绘图展示
def cv_show(img,name='undefine'):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#读入图片
image = cv2.imread("./images/timg6.jpg")
#显示图片
cv_show(image);
#识别结果
ret = HyperLPR_PlateRecogntion(image)
print(ret)