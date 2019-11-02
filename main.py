import cv2
import matplotlib.pyplot as plt
# 绘图展示
def cv_show(img,name='undefine'):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def plt_show(img):
	plt.imshow(img)
	plt.show()

# 读取一个模板图像
img = cv2.imread('./images/timg1.jpg')
cv_show(img)
# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(gray)

# 二值图像
gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
cv_show(gray)

# 截取
cut = gray[150:250,75:350]
cv_show(cut)

# 边缘检测
gray = cv2.Canny(gray, 50, 100)
cv_show(gray)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray,refCnts,-1,(0,0,255),3)
cv_show(gray)
