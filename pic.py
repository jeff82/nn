import PIL
from PIL import Image
from PIL import ImageFilter

dir=r'E:\py\nn\dcfls\fuck pic\subpic'
name='\grid12xex.jpg'
img = Image.open(dir+name)  
#Image._show(img)
# rgb2xyz = (
#    0.412453, 0.357580, 0.180423, 0,
#    0.212671, 0.715160, 0.072169, 0,
#    0.019334, 0.119193, 0.950227, 0 )
#out = img.convert("RGB", rgb2xyz)
im= img.filter(ImageFilter.SMOOTH)
im=img.filter(ImageFilter.Kernel((3,3),(1,1,1,0,0,0,2,0,2)))
#im.show()
w=10
h=10
pixel = im.getpixel((w, h))   