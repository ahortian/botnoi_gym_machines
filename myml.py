
# pip install flask
# pip install botnoi==0.2.1
# pip install requests, numpy, Pillow, 
# sklearn, keras, tensorflow, pandas

from botnoi import cv
import pickle


# def main():
#     # Load Model
#     modFile = 'gymmachine.mod'
#     mod = pickle.load(open(modFile,'rb'))
#     # input image
#     pic_name = 'testimage.jpg'
#     test_image = cv.image(pic_name)
#     feat = test_image.getmobilenet()
#     res = mod.predict([feat])
#     #print(res)
#     #rint(type(res))
#     print(res[0])

# if __name__ == '__main__':
#     main()

# Create pipeline
def predictImage(pic_name):
    #Load model
    modFile = 'gymmachine.mod'
    mod = pickle.load(open(modFile,'rb'))
    # input image
    test_image = cv.image(pic_name)
    feat = test_image.getmobilenet()
    res = mod.predict([feat])
    #print(res)
    return res[0]

#print('The result is: {}.'.format(predictImage('testimage.jpg')))