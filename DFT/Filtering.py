# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import cv2 as cv
import numpy as np
import math

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order

    def distancematrix(self, shape):
        distanceMatrix = np.zeros(shape)
        u = 0
        while u < shape[0]:
            v = 0
            while v < shape[1]:
                distancevalue = math.sqrt(math.pow((u - shape[0] / 2), 2) + math.pow((v - shape[1] / 2), 2))
                distanceMatrix[u][v] = distancevalue
                v += 1
            u += 1
        return distanceMatrix

    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        distanceMatrix =self.distancematrix(shape)
        print(shape, cutoff)
        u = 0
        while u < shape[0]:
            v = 0
            while v < shape[1]:
                if distanceMatrix[u][v] <= cutoff:
                    distanceMatrix[u][v] = 255
                else:
                    distanceMatrix[u][v] = 0
                v += 1
            u += 1
        cv.imwrite('idealLow.png',distanceMatrix)
        return distanceMatrix


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        distanceMatrix =  255*np.ones(shape) - self.get_ideal_low_pass_filter(shape, cutoff)

        cv.imwrite('idealHigh.png', distanceMatrix)
        return distanceMatrix

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        distanceMatrix = self.distancematrix(shape)
        u = 0
        while u < shape[0]:
            v = 0
            while v < shape[1]:
                distanceMatrix[u][v] = (255/(1 + math.pow((distanceMatrix[u][v]/cutoff), 2 * order)))
                v += 1
            u += 1
        cv.imwrite('aa.png', distanceMatrix)
        return distanceMatrix

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        distanceMatrix = self.distancematrix(shape)
        u = 0
        while u < shape[0]:
            v = 0
            while v < shape[1]:
                if distanceMatrix[u][v] ==0:
                    distanceMatrix[u][v] =0
                else:
                    distanceMatrix[u][v] = (255 / (1 + math.pow((cutoff/distanceMatrix[u][v]), 2 * order)))
                v += 1
            u += 1
        cv.imwrite('butterHigh.png', distanceMatrix)
        return distanceMatrix

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        distanceMatrix = self.distancematrix(shape)
        u = 0
        while u < shape[0]:
            v = 0
            while v < shape[1]:
                distanceMatrix[u][v] = 255 * math.pow(math.e, - math.pow(distanceMatrix[u][v], 2) / (2 * math.pow(cutoff, 2)))
                v += 1
            u += 1
        cv.imwrite('gaussian.png', distanceMatrix)
        return distanceMatrix


    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        OnesMatrix = 255 * np.ones(shape)
        highPassFilter = OnesMatrix - self.get_gaussian_low_pass_filter(shape, cutoff)
        cv.imwrite('gaussianHigh.png', highPassFilter)
        return highPassFilter

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        A = image.min()
        B = image.max()
        i = 0
        while i < len(image):
            j = 0
            while j < len(image[i]):
                image[i][j] = (255/(B-A))*(image[i][j] - A)
                j += 1
            i += 1

        return image

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        print((self.image).shape)
        fftImage = np.fft.fft2(self.image)
        fftCenter = np.fft.fftshift(fftImage)
        if self.filter == self.get_butterworth_low_pass_filter or self.filter == self.get_butterworth_high_pass_filter:
            mask = self.filter(fftImage.shape, self.cutoff, self.order)
        else:
            mask = self.filter(fftImage.shape, self.cutoff)
        filteredImage = fftCenter * mask;
        filteredImageInveseShift = np.fft.ifftshift(filteredImage)
        inverseFourier = np.fft.ifft2(filteredImageInveseShift)
        fullContractStrech = np.uint8(self.post_process_image(np.absolute(inverseFourier)));
        magnitudeOfDFT = np.absolute(fftCenter)
        magnitudeOfDFT = self.post_process_image(np.uint8(np.log(magnitudeOfDFT)))
        magnitudeOfInverse = self.post_process_image(np.uint8(np.log(1 + np.absolute(filteredImage))))
        return [fullContractStrech, magnitudeOfDFT, magnitudeOfInverse]