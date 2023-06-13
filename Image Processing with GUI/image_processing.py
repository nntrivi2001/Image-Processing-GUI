# Nguyễn Ngọc Trí Vĩ - Image Processing Course work


from tkinter import font, filedialog, StringVar
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfile
import cv2
import numpy as np

import os



class App(Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.defaultFont = font.nametofont("TkDefaultFont")
        self.defaultFont.configure(family = "Calibri", size = 16, weight = font.BOLD)
        self.place(relx = 0.5, rely = 0.5, anchor = "center")
        self.pack()

def open_file():
   file = filedialog.askopenfile(mode='r', filetypes=[('PNG', '*.png'), ('JPG', '*.jpg')])
   if file:
        global filepath
        filepath = os.path.abspath(file.name)
        filepath = filepath.replace("\\", "\\\\")
        img = show_picture(filepath)
        opened_file = Label(frm, image = img)
        opened_file.grid(column=1, row=1)
        opened_file.image = img
        global cv_img
        cv_img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)

def open_filepath():
    global folder
    folder = filedialog.askdirectory(title="select").replace("/", "\\\\")
    if folder:
        Label(frm, text = folder).grid(column=2, row=5, columnspan = 3)
        

def show_picture(path):
    img = ImageTk.PhotoImage((Image.open(path)).resize((400,300)))
    return img

def callback(temp1, temp2, temp3):
    K = input.get()

    if(K.isdigit()):
        global k
        k = int(float(K))

def func_LamNetAnh():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    anhnet = cv2.filter2D(cv_img, -1, kernel)

    cv2.imwrite(".\\sharpened.png", anhnet)

    sharpened_img = show_picture(".\\sharpened.png")
    sharpened = Label(frm, image = sharpened_img)
    sharpened.grid(column=3, row=1)
    sharpened.image = sharpened_img

def func_LamMoAnh():
    # ### Dùng bộ lọc thông thấp trung vị
    # anhmo = cv2.medianBlur(cv_img, k)

    ## Dùng xử lý ảnh hình thái (mở ảnh để xử lý)
    kernel = np.ones((k, k),np.uint8)
    open_img = cv2.morphologyEx(cv_img, cv2.MORPH_OPEN, kernel)
    anhmo = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
    anhmo = anhmo.astype(np.uint8)

    cv2.imwrite(".\\blurred.png", anhmo)

    blurred_img = show_picture(".\\blurred.png")
    blurred = Label(frm, image = blurred_img)
    blurred.grid(column=3, row=1)
    blurred.image = blurred_img

def func_PhanDoanAnh():
    ### Dùng K-Means
    if(len(cv_img.shape) < 3):
        Z = np.float32(cv_img.reshape((-1, )))
    else:
        Z = np.float32(cv_img.reshape((-1, 3)))
    
    # TERM_CRITERIA_EPS : ngừng khi đạt được giá trị epsilon
    # TERM_CRITERIA_MAX_ITER:  ngừng khi đạt số lần lặp tối đa
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Áp dụng K-Means
    ret,label,center=cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Chuyển về dạng uint8 sau đó chuyển thành form ảnh ban đầu
    res = np.uint8(center)[label.flatten()]
    anhphandoan = res.reshape((cv_img.shape))

    cv2.imwrite(".\\segmentated.png", anhphandoan)

    segmentated_img = show_picture(".\\segmentated.png")
    segmentated = Label(frm, image = segmentated_img)
    segmentated.grid(column=3, row=1)
    segmentated.image = segmentated_img

def func_TrichBienAnh():

    ## Làm bằng xử lý ảnh hình thái 
    kernel = np.ones((k, k),np.uint8)
    erode_img = cv2.erode(cv_img, kernel)
    bienanh = cv_img - erode_img
    bienanh = bienanh.astype(np.uint8)

    # ## Dùng bộ lọc Laplacian
    # bienanh = cv2.Laplacian(cv_img, cv2.CV_64F)

    # ## Dùng bộ lọc Sobel
    # bienanh = cv2.Sobel(cv_img, cv2.CV_64F, 1, 1, ksize=k)

    # ## Dùng Canny
    # bienanh = cv2.Canny(cv_img, 100, 200)

    cv2.imwrite(".\\boundary_extracted.png", bienanh)
    
    boundary_extracted_img = show_picture(".\\boundary_extracted.png")
    boundary_extracted = Label(frm, image = boundary_extracted_img)
    boundary_extracted.grid(column=3, row=1)
    boundary_extracted.image = boundary_extracted_img

def func_TrichXuatDacTrung():
    Hog = cv2.HOGDescriptor()
    h = Hog.compute(cv2.resize(cv_img, (128, 128)))
    np.savetxt(".\\hog_img.txt", h)
    os.startfile(".\\hog_img.txt")

def func_LamNetBoDuLieu():
    ## Làm bằng xử lý ảnh hình thái 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    if(os.path.exists("sharpen") == False):
        os.mkdir("sharpen")

    for img_path in os.listdir(folder):
        img_fullpath = f'{folder}\\{img_path}'
        img = cv2.imread(img_fullpath, cv2.IMREAD_ANYCOLOR)
        
        anhnet = cv2.filter2D(img, -1, kernel)

        cv2.imwrite(f'sharpen/{img_path}', anhnet)

def func_LamMoBoDuLieu():
    ## Làm bằng xử lý ảnh hình thái 
    kernel = np.ones((k, k),np.uint8)

    if(os.path.exists("blur") == False):
        os.mkdir("blur")

    for img_path in os.listdir(folder):
        img_fullpath = f'{folder}\\{img_path}'
        img = cv2.imread(img_fullpath, cv2.IMREAD_ANYCOLOR)
        
        ### Dùng bộ lọc thông thấp trung vị
        anhmo = cv2.medianBlur(img, k)

        ### Dùng xử lý ảnh hình thái (mở ảnh để xử lý)
        # open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # anhmo = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
        # anhmo = anhmo.astype(np.uint8)

        cv2.imwrite(f'blur/{img_path}', anhmo)

def func_PhanDoanBoDuLieu():
    if(os.path.exists("segmentation") == False):
        os.mkdir("segmentation")

    for img_path in os.listdir(folder):
        img_fullpath = f'{folder}\\{img_path}'
        img = cv2.imread(img_fullpath, cv2.IMREAD_ANYCOLOR)
        
        ### Dùng K-Means
        if(len(img.shape) < 3):
            Z = np.float32(img.reshape((-1, )))
        else:
            Z = np.float32(img.reshape((-1, 3)))
        
        # TERM_CRITERIA_EPS : ngừng khi đạt được giá trị epsilon
        # TERM_CRITERIA_MAX_ITER:  ngừng khi đạt số lần lặp tối đa
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Áp dụng K-Means
        ret,label,center=cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Chuyển về dạng uint8 sau đó chuyển thành form ảnh ban đầu
        res = np.uint8(center)[label.flatten()]
        anhphandoan = res.reshape((img.shape))

        cv2.imwrite(f'segmentation/{img_path}', anhphandoan)

def func_TrichBienBoDuLieu():
    ## Làm bằng xử lý ảnh hình thái 
    kernel = np.ones((k, k),np.uint8)

    if(os.path.exists("boundary_extraction") == False):
        os.mkdir("boundary_extraction")

    for img_path in os.listdir(folder):
        img_fullpath = f'{folder}\\{img_path}'
        img = cv2.imread(img_fullpath, cv2.IMREAD_ANYCOLOR)
        
        erode_img = cv2.erode(img, kernel)
        bienanh = img - erode_img
        bienanh = bienanh.astype(np.uint8)

        ### Dùng bộ lọc Laplacian
        # bienanh = cv2.Laplacian(img, cv2.CV_64F)

        ### Dùng bộ lọc Sobel
        # bienanh = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=k)

        ### Dùng Canny
        #bienanh = cv2.Canny(img, 100, 200)

        cv2.imwrite(f'boundary_extraction/{img_path}', bienanh)




# Tạo ứng dụng
myapp = App()

myapp.master.title("Đồ án môn học Thực hành Xử lý ảnh")
myapp.master.minsize(1000, 400)
myapp.master.maxsize(1500, 800)

frm = Frame(myapp, padding = 24)
frm.grid()

cv_img = cv2.imread('.\\alert.png', cv2.IMREAD_ANYCOLOR)

Button(frm, text="Click để nhập ảnh", command = open_file).grid(column = 1, row = 0, pady = 10, ipady = 4, ipadx = 16)
Label(frm, text="Ảnh sau khi được xử lý").grid(column = 3, row = 0, pady = 10)

Label(frm, text = "Nơi thể hiện ảnh nhập vào", background = "white", anchor="center").grid(column=1, row=1, ipadx = 85, ipady = 137)
Label(frm, text = "Nơi thể hiện ảnh sau xử lý", background = "white", anchor="center").grid(column=3, row=1, ipadx = 85, ipady = 137)

Label(frm, text="K (K x K)/Size Kernal").grid(column = 2, row = 2)

K = StringVar()
K.trace("w", callback)

input = Entry(frm, textvariable = K)
input.grid(column = 2, row = 3, ipady = 4, ipadx = 4)

Button(frm, text="Làm nét ảnh", command = func_LamNetAnh).grid(column = 0, row = 4, pady = 16, ipady = 12, ipadx = 16)
Button(frm, text="Làm mờ ảnh", command = func_LamMoAnh).grid(column = 1, row = 4, ipady = 12, ipadx = 16)
Button(frm, text="Phân đoạn ảnh", command = func_PhanDoanAnh).grid(column = 2, row = 4, ipady = 12, ipadx = 16)
Button(frm, text="Trích biên ảnh", command = func_TrichBienAnh).grid(column = 3, row = 4, ipady = 12, ipadx = 16)
Button(frm, text="Trích xuất đặc trưng", command = func_TrichXuatDacTrung).grid(column = 4, row = 4, ipady = 12, ipadx = 16)

Button(frm, text="Chọn đường dẫn", command = open_filepath).grid(column = 0, row = 5, pady = 10, ipady = 4, ipadx = 32, columnspan = 2)

Button(frm, text="Làm nét bộ dữ liệu", command = func_LamNetBoDuLieu).grid(column = 0, row = 6, pady = 16, ipady = 12, ipadx = 16)
Button(frm, text="Làm mờ bộ dữ liệu", command = func_LamMoBoDuLieu).grid(column = 1, row = 6, ipady = 12, ipadx = 16)
Button(frm, text="Phân đoạn bộ dữ liệu", command = func_PhanDoanBoDuLieu).grid(column = 2, row = 6, ipady = 12, ipadx = 16)
Button(frm, text="Trích biên bộ dữ liệu", command = func_TrichBienBoDuLieu).grid(column = 3, row = 6, ipady = 12, ipadx = 16)
Button(frm, text="Thoát", command = myapp.quit).grid(column = 4, row = 6, ipady = 12, ipadx = 16)

# start the program
myapp.mainloop()