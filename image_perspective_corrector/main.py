import cv2
import time
import numpy as np

# 全局变量
src_points = []  # 存储四个映射点坐标
dragging = -1    # 当前拖动的点索引
radius = 15      # 控制点判定半径

# 现象：矫正图像出现镜像翻转
# 解决：强制按左上→右上→右下→左下顺序排序
def order_points(pts):
    rect = np.zeros((4,2), dtype="int32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上 
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下
    return rect

def get_four_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)   
    
    if len(approx) == 4:
        src_pts = order_points(approx.reshape(4,2)).astype(np.int32)
        return order_points(src_pts)    
        # h = int(np.linalg.norm(src_pts[0]-src_pts[3]))
        # w = int(np.linalg.norm(src_pts[0]-src_pts[1]))
        # dst_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])        
        # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # corrected = cv2.warpPerspective(img, M, (w, h))
        # return corrected
    else:
        print("未检测到四边形轮廓")
        return None

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global src_points, dragging
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查是否点击到控制点
        for i in range(4):
            if np.linalg.norm(np.array([x,y]) - np.array(src_points[i])) < radius:
                dragging = i
                break
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging != -1:
        # 更新当前拖动点坐标
        # 在mouse_callback函数中加入限制
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        src_points[dragging] = (x, y)
        update_preview()
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

# 更新透视预览
def update_preview():    
    # 定义目标点（固定矩形）
    w, h = src_points[2][0] - src_points[0][0], src_points[2][1] - src_points[0][1]
    print(f"宽x高 = {w}x{h}")    
    # dst_pts = np.float32([[0, 0], [h, 0], [h, w], [0, w]])
    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(np.float32(src_points), dst_pts)
    
    # 执行变换    
    warped = cv2.warpPerspective(original_img, M, (w, h))
    
    # 绘制控制点
    preview = original_img.copy()
    for pt in src_points:
        cv2.circle(preview, pt, radius, (0,255,0), -1)
    
    # 显示实时效果
    cv2.imshow('Original', preview)
    cv2.imshow('Warped', cv2.resize(warped, (w//2, h//2)))
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 按下s键保存
            # 按时间戳保存当前矫正结果
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            cv2.imwrite(f'{img_path}_{timestamp}.jpg', warped)
            print("当前矫正结果已保存！")
        elif key == 27:  # ESC键退出
            break

if __name__ == "__main__":
    # 读取图像
    img_path = input("请输入图片路径（123.jpg）：")
    if img_path.strip() == "":
        img_path = '123.jpg'
    original_img = cv2.imread(img_path)

    if original_img is None:
        print("Error: Image not found!")
        exit(1)    
    h, w = original_img.shape[:2]
        
    src_points = get_four_points(original_img)  
    if src_points is None:
        # 初始化四个角点（可自定义初始位置）
        src_points = [(w//4, h//4), (3*w//4, h//4), (3*w//4, 3*h//4), (w//4, 3*h//4)]
    print(f"初始点位置：{src_points}\n拖拽调整矫正点位置，按下s键保存矫正文件为'corrected_{{timestamp}}.jpg'，ESC键退出：")
        
    # 创建窗口
    cv2.namedWindow('Original')
    cv2.setMouseCallback('Original', mouse_callback)
    
    # 初始显示
    update_preview()
    cv2.waitKey(0)
    cv2.destroyAllWindows()