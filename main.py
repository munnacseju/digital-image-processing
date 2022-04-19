# from matplotlib import pyplot as plt
# # from skimage import io
# import scipy.misc as ms
#
# import numpy as np
# import scipy.misc as ms
# from PIL import Image
#
# import matplotlib.pyplot as plt
# import skimage.io as sk
# img = sk.imread("img.jpg")
# plt.imshow(img*[0, 0, 1])
# plt.show()
# print(type(img))

#print(img*[0., 0., 1])

#
# image = Image.open("img.jpg")
# print(type(image))
# img = np.asarray(image)
# # print(img*[0., 0., 1])
# # print(img*[0., 0., 1])
# img = img*[0., 0., 1]
# print(type(img))
# # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11]])
# # print(a)
# # print(a[0:3, :3] )
# #
# # a = np.zeros((3,3))
# # print(a)
# # a = np.eye(3)
# # print(a)
# #
# # a = np.random.random((2,2))
# # print(a)
#
# # a = np.array([[1,2],[3,4]])
# # b = np.array([[5,6],[7,8]])
# # c = a + b
# # print(c)


# import matplotlib.image as IMG
# import matplotlib.pyplot as plt
# img = IMG.imread("img.jpg")
# print(img.shape)
# img = img * [0]
# img = img + [250, 0, 255]
# print(img)
# plt.imshow(img)
# plt.colorbar()
# plt.show()

# import  cv2
# from matplotlib.pyplot import plot as plt
#
# grey_image = cv2.imread("img.jpg", 0)
# color_image = cv2.imread("img.jpg", 1)
#
# cv2.imshow("Grey Image", grey_image)
# cv2.imshow("Color Image", color_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindow()
from PIL import Image
import matplotlib.pyplot as plt
mankey = Image.open("mankey.jpg")
image = Image.open("img.jpg")

# plt.subplot(1,2,2)
# plt.imshow(image)
# # img = image.resize((100, 100))
# img = image.crop((100, 0, 400, 300))
# plt.subplot(1, 2, 1)
# plt.imshow(img)

image_copy = image.copy()
image_copy.paste(mankey, (210, 210))
plt.imshow(image_copy)
plt.show()
# from PIL import Image
# img = Image.open("img.jpg")
# print(img.size)
#
# # tmImage = img.thumbnail((1200, 1300))
# # tmImage.save("test/thumbnail_image.jpg")
#
# smImage = img.resize((200, 300))
# smImage.save("test/_small_image.jpg")
#
# cropImage = img.crop((0, 0, 300, 300))
# cropImage.save("test/crop_image.jpg")

# from PIL import Image
# img1 = Image.open("img/img.jpg")
# img2 = Image.open("img/mankey.jpg")

# copy_img = img2.copy()
# copy_img.paste(img1, (100, 100))
#
# copy_img.save("img/copy.jpg")
#
# rotate_img = img1.rotate((180))
# rotate_img.save("img/rotate.jpg")
#
# img2flip = img2.transpose(Image.FLIP_LEFT_RIGHT)
# img2flip.save("img/flip_image.jpg")
#
# from skimage import io
# from matplotlib import pyplot as plt
# img = io.imread("img/img.jpg", as_gray=True)
# from skimage.transform import rescale, resize, downscale_local_mean
# rescaledImage = rescale(img,1.0/4.0, anti_aliasing=True)
# resized_img = resize(img, (200, 200))
# downscaled_img = downscale_local_mean(img, (4,3))
# plt.subplot(2, 2, 1)
# plt.title("Rescaled image")
# plt.imshow(rescaledImage)
#
# plt.subplot(2, 2, 2)
# plt.title("Resized image")
# plt.imshow(resized_img)
#
# plt.subplot(2, 2, 3)
# plt.title("Downscaled image")
# plt.imshow(downscaled_img)
#
# plt.subplot(2, 2, 4)
# plt.title("Original image")
# plt.imshow(rescaledImage)
# plt.show()

# from skimage import io
# from matplotlib import pyplot as plt
# img = io.imread("img/img.jpg", as_gray=True)
# from skimage import restoration
# import numpy as np
# psf = np.ones((3,3))/9
# deconvolved, _ = restoration.unsupervised_wiener(img, psf)
#
# plt.imsave("img/deconvolved.jpg", deconvolved)

# import  matplotlib.pyplot as plt
# from skimage import io, restoration
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# img = io.imread("img/ski.png")
# entr_img = entropy(img, disk(3))
# plt.imshow(entr_img)
from  skimage.filters import try_all_threshold

# print("Hello")