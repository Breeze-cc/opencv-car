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

def change_size(img,rate):
    height, width = img.shape[:2]
    size = (int(width * rate), int(height * rate))
    img = cv2.resize(img, size)
    return img

def ocr(img):
    gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show(gray,'ocr')
    pytesseract.pytesseract.tesseract_cmd = 'C://Program Files/Tesseract-OCR/tesseract.exe'
    result = pytesseract.image_to_string(gray, lang='chi_sim')  # 简体中文=chi_sim
    return result

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel7x7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# 读取一个图像
img = cv2.imread('./images/timg1.jpg')
cv_show(img)

# 缩放图像
img = change_size(img,1)
cv_show(img)

# 截取
# plt_show(img)
cut = img[0:90,10:100]
cv_show(cut)

# 灰度图
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(grayImg)

# 二值图像
gray = cv2.threshold(grayImg, 200, 255, cv2.THRESH_BINARY)[1]
cv_show(gray)

# 开运算
grayOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
cv_show(grayOpen,'open')


gray = grayOpen.copy()

# 膨胀
for i in range(5):
	gray = cv2.dilate(gray, kernel7x7)
	cv_show(gray, 'dilate')


refCnts, hierarchy = cv2.findContours(gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img.copy(),refCnts,-1,(0,0,255),2)
cv_show(res,'contour')

img2 = img.copy()
locs = []
for i in range(0,len(refCnts)):
    x, y, w, h = cv2.boundingRect(refCnts[i])
    img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 2)
    # 根据坐标提取每一个组
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来 440/140
    if ar > 2.8 and ar < 3.5:
        # 符合的留下来
        locs.append((x, y, w, h))
        img3 = grayOpen[y:y + h, x:x + w]
        cv_show(img3, str(i))
        print(ocr(img3))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
print(locs)
cv_show(img2,'rectangle')
exit()


# 开运算
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
cv_show(gray,'open')

# 闭运算
morphClose = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKernel)
cv_show(morphClose,'close')
exit()

# 腐蚀
gray = cv2.erode(gray,rectKernel)
cv_show(gray,'erode')

# 膨胀
gray = cv2.dilate(gray,rectKernel)
cv_show(gray,'dilate')

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




