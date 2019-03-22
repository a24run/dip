# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
import math
class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        resultMatrix = np.zeros(matrix.shape, dtype = complex)
        N = matrix.shape[1];
        u = 0
        while u < len(resultMatrix):
            v = 0
            while v < len(resultMatrix[u]):
                i = 0
                while i < len(matrix):
                    j = 0
                    while j < len(matrix[i]):
                        resultMatrix[u][v] += matrix[i][j] * ((math.cos(((2*math.pi)/N)*(u*i + v*j))) - 1j * math.sin(((2*math.pi)/N)*(u*i + v*j)))
                        j += 1
                    i += 1
                v += 1
            u += 1
        return resultMatrix


    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        resultMatrix = np.zeros(matrix.shape, dtype=complex)
        value = 0;
        N = matrix.shape[1];
        i = 0
        while i < len(resultMatrix):
            j = 0
            while j < len(resultMatrix[i]):
                u = 0
                while u < len(matrix):
                    v = 0
                    while v < len(matrix[u]):
                        resultMatrix[i][j] += matrix[u][v] * ((math.cos(((2 * math.pi) / N) * (u * i + v * j))) + 1j * math.sin(((2 * math.pi) / N) * (u * i + v * j)))
                        v += 1
                    u += 1
                j += 1
            i += 1
        return resultMatrix


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        resultMatrix = np.zeros(matrix.shape)
        value = 0;
        N = matrix.shape[1];
        u = 0
        while u < len(resultMatrix):
            v = 0
            while v < len(resultMatrix[u]):
                i = 0
                while i < len(matrix):
                    j = 0
                    while j < len(matrix[i]):
                        resultMatrix[u][v] += matrix[i][j] * ((math.cos(((2 * math.pi) / N) * (u * i + v * j))) )
                        j += 1
                    i += 1
                v += 1
            u += 1
        return resultMatrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        resultMatrix = self.forward_transform(matrix)
        solutionMatrix = np.zeros(resultMatrix.shape)
        i=0
        while i < len(resultMatrix):
            j = 0
            while j < len(resultMatrix[i]):
                solutionMatrix[i][j] = math.sqrt(math.pow(np.real(resultMatrix[i][j]),2) + math.pow(np.imag(resultMatrix[i][j]),2))
                j += 1
            i+=1
        return solutionMatrix
