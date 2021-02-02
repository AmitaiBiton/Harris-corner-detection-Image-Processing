
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize, binary_closing
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
def filterGS(img , filter):
    if filter.shape[0]==3:
        temp = np.zeros((425, 421))
        for i in range(421):
            for j in range(417):
                for l in range(len(filter[0])):
                    t = 0
                    for k in range(len(filter[0])):
                        t += img[i + l][k + j] * filter[k][l]
                    temp[i][j] += t
        return temp
    # output matrix after the multiply
    elif filter[0].shape !=3:
        temp = np.zeros((image.shape[0]+4,image.shape[1]+4))
        for i in range(418):
            for j in range(415):
                for  l in range(len(filter[0])):
                    t = 0
                    for k in range (len(filter[0])):
                         #print(i+l , k+j , image.shape)
                         t += img[i+l][k+j]*filter[k][l]
                    temp[i][j] += t
        return temp
def gaussians(image):

    Gaussians = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256

    clean_image = cv2.filter2D(image,cv2.CV_64F, Gaussians)
    #afterClean = clean_image[2:425, 2:421]
    #result = afterClean[:423, :419]
    return clean_image


def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def round_angle(angle):
    angle = np.rad2deg(angle)%180
    if (0<=angle<22.5) or (157.5<=angle<180):
        angle=0
    elif 22.5 <= angle < 67.5:
        angle=45
    elif 67.5 <=angle < 112.5:
        angle=90
    elif 112.5<=angle<157.5:
        angle=135
    return  angle

def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]

            except IndexError as e:
                pass

    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }
    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)
    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M-1):
        for j in range(N-1):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                    or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                    or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def hough_lines_draw(img, indicies, rhos, thetas):
    list  = []
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if indicies[i][1] == 0 and i == len(indicies) - 1:
            y1 -= 25
            y2 += 25

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

def getMax(H, num_peaks, threshold=0, nhood_size=3):

    # loop through number of peaks to identify
    list = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        list.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return list, H


if  __name__ == "__main__":

    image = cv2.imread('sudoku-original.jpg ', cv2.IMREAD_GRAYSCALE)
    gasImg = gaussians(image)
    img2, D = gradient_intensity(gasImg)
    img3 = suppression(np.copy(img2), D)
    img4, weak = threshold(np.copy(img3), 80, 130)
    img5 = tracking(np.copy(img4), weak)

    accumulator, thetas, rhos = hough_line(img5)
    indi, h = getMax(accumulator, 8, nhood_size=200)
    img = np.zeros((423, 419))
    hough_lines_draw(img, indi, rhos, thetas)

    kernel = np.ones((3, 3), dtype=np.uint8) / 9
    img  = cv2.filter2D(img ,-1 , kernel)
    #img = gaussians(img)
    ratio = [0.005]
    count = 0  # for equivalent ratio access

    # Apply gaussian blurring
    blur_img = gaussians(img)

    fx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    fy  =  np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    # Find gradient Fx
    x_grad = filterGS(blur_img , fx)

    # Find gradient Fy
    y_grad = filterGS(blur_img , fy)

    xx_grad = x_grad * x_grad
    yy_grad = y_grad * y_grad
    xy_grad = x_grad * y_grad
    tuple_data = []  # Contains y, x Co-ordinates and its corner response
    k = 0.04
    max = 0.01
    for i in range(0, int(img.shape[0])-1):
        for j in range(0, int(img.shape[1])-1):
            window_x = xx_grad[i - 4: i + 5, j - 4: j + 5]
            window_y = yy_grad[i - 4: i + 5, j - 4: j + 5]
            window_xy = xy_grad[i - 4: i + 5, j - 4: j + 5]
            sum_xx = np.sum(window_x)
            sum_yy = np.sum(window_y)
            sum_xy = np.sum(window_xy)
            determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
            trace = sum_xx + sum_yy
            R = determinant - (k * trace * trace)
            tuple_data.append((i, j, R))
            if (R > max):
                max = R

    L = []
    thres_ratio = ratio[count]
    count += 1
    threshold = thres_ratio * max
    for res in tuple_data:
        i, j, R = res
        if R > threshold:
            L.append([i, j, R])
    # Phase III : Non maximal suppression
    sorted_L = sorted(L, key=lambda x: x[2], reverse=True)
    final_L = []  # final_l contains list after non maximal suppression
    final_L.append(sorted_L[0][:])

    dis = 10
    xc, yc = [], []
    for i in sorted_L:
        for j in final_L:
            if (abs(i[0] - j[0] < dis) and abs(i[1] - j[1]) < dis):
                break
        else:
            final_L.append(i[:-1])
            xc.append(i[1])
            yc.append(i[0])


    plt.subplot(1,2,1), plt.imshow(image, cmap="gray"), plt.title('image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(image, cmap="gray"), plt.title('gaussian')
    plt.xticks([]), plt.yticks([])
    plt.plot(xc,yc,'*', color='red')

    plt.show()
