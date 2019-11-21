import matplotlib.pyplot as plt
import pytesseract
import cv2

# 绘图展示
def cv_show(img,name='undefine'):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def plt_show(img):
    plt.imshow(img)
    plt.show()

def ocr(img):
    gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show(gray,'ocr')
    pytesseract.pytesseract.tesseract_cmd = 'C://Program Files/Tesseract-OCR/tesseract.exe'
    result = pytesseract.image_to_string(gray, lang='chi_sim')  # 简体中文=chi_sim
    return result

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# 读取一个图像
img = cv2.imread('./images/ifchange.png')
cv_show(img)

# 缩放图像
rate = 0.5
height, width = img.shape[:2]
size = (int(width*rate), int(height*rate))
img = cv2.resize(img,size)
cv_show(img)


# 缩放图像
rate = 0.5
height, width = img.shape[:2]
size = (int(width*rate), int(height*rate))
img = cv2.resize(img,size)
cv_show(img)

# plt_show(img)
# 截取
cut = img[0:90,10:100]
cv_show(cut)

# 灰度图
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(grayImg)
# 二值图像
gray = cv2.threshold(grayImg, 200, 255, cv2.THRESH_BINARY_INV)[1]
cv_show(gray)
exit()

# 开运算
grayOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
cv_show(grayOpen,'open')

gray = grayOpen.copy()
# 膨胀
for i in range(5):
	gray = cv2.dilate(gray, kernel9x9)
	cv_show(gray, 'dilate')

refCnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img.copy(),refCnts,-1,(0,0,255),2)
cv_show(res,'contour')

img2 = img.copy()
locs = []
for i in range(0,len(refCnts)):
    x, y, w, h = cv2.boundingRect(refCnts[i])
    img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 2)
    # 根据坐标提取每一个组
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.8 and ar < 3.5:
        # 符合的留下来
        locs.append((x, y, w, h))
        group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
        cv_show(group, 'i')
        img3 = grayOpen[y - 5:y + h + 5, x - 5:x + w + 5]
        cv_show(img3, 'i')
        print(ocr(img3))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
print(locs)
cv_show(img2,'rectangle')


exit()


# 开运算
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
cv_show(gray,'open')

refCnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img.copy(),refCnts,-1,(0,0,255),2)
cv_show(res,'contour')

# 闭运算
morphClose = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKernel)
cv_show(morphClose,'close')
exit()

# 膨胀
gray = cv2.dilate(gray,rectKernel)
cv_show(gray,'dilate')

# 腐蚀
gray = cv2.erode(gray,rectKernel)
cv_show(gray,'erode')

# 开操作
morphOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
cv_show(gray,'open')

# 二值图像
gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
cv_show(gray)


img2 = gray.copy()



pytesseract.pytesseract.tesseract_cmd = 'C://Program Files/Tesseract-OCR/tesseract.exe'
result =pytesseract.image_to_string(img2,lang='chi_sim') # 简体中文=chi_sim
print(result)
exit()

for i in range(5):
	gray = cv2.dilate(img2, kernel9x9)
	cv_show(gray, 'dilate')


# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray,refCnts,-1,(0,0,255),3)
cv_show(gray)


for i in range(0,len(refCnts)):
	x, y, w, h = cv2.boundingRect(refCnts[i])
	cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 5)
	# 根据坐标提取每一个组
	group = gray[y - 5:y + h + 5, x - 5:x + w + 5]
	cv_show(group,'i')

# # 开操作
# morphOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
# cv_show(gray,'open')
#
# # 闭操作
# morphClose = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKernel)
# cv_show(gray,'close')
#
# # 礼帽
# tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show(tophat,'tophat')
#
# # 黑帽
# blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# cv_show(blackhat,'blackhat')


# 边缘检测
gray = cv2.Canny(gray, 50, 100)
cv_show(gray)




