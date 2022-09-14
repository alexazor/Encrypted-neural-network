from gmpy2 import mpz
import numpy as np


class Q_int():
    def __init__(self, val, q_factor):
        self.val = mpz(val)
        self.q = mpz(q_factor)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Q_int(self.val + other.val, self.q)
        elif (isinstance(other, mpz)):
            return Q_int(self.val + other, self.q)
        elif (isinstance(other, int)):
            return Q_int(self.val + self.q*other, self.q)
        elif (isinstance(other, float)):
            return Q_int(self.val + mpz(self.q*other), self.q)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return Q_int(self.val - other.val, self.q)
        elif (isinstance(other, mpz)):
            return Q_int(self.val - other, self.q)
        elif (isinstance(other, int)):
            return Q_int(self.val - self.q*other, self.q)
        elif (isinstance(other, float)):
            return Q_int(self.val - mpz(self.q*other), self.q)
        else:
            raise TypeError(
                "unsupported operand type(s) for -: '{}' and '{}'".format(self.__class__, type(other)))

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return Q_int(other.val - self.val, self.q)
        elif (isinstance(other, mpz)):
            return Q_int(other - self.val, self.q)
        elif (isinstance(other, int)):
            return Q_int(self.q*other - self.val, self.q)
        elif (isinstance(other, float)):
            return Q_int(mpz(self.q*other) - self.val, self.q)
        else:
            raise TypeError(
                "unsupported operand type(s) for - (reverse): '{}' and '{}'".format(self.__class__, type(other)))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Q_int((self.val * other.val)//self.q, self.q)
        elif (isinstance(other, mpz)):
            return Q_int((self.val * other)//self.q, self.q)
        elif (isinstance(other, int)):
            return Q_int(self.val * other, self.q)
        elif (isinstance(other, float)):
            return Q_int(mpz(self.val * other), self.q)
        elif(isinstance(other, np.ndarray)):
            return self.val*other
        else:
            raise TypeError(
                "unsupported operand type(s) for *: '{}' and '{}'".format(self.__class__, type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, self.__class__):
            return Q_int(mpz(self.q*self.val / other.val), self.q)
        elif (isinstance(other, mpz)):
            return Q_int(mpz(self.q*self.val / other), self.q)
        elif (isinstance(other, int)):
            return Q_int(mpz(self.val / other), self.q)
        elif (isinstance(other, float)):
            return Q_int(mpz(self.val / other), self.q)
        else:
            raise TypeError(
                "unsupported operand type(s) for //: '{}' and '{}'".format(self.__class__, type(other)))

    def __rfloordiv__(self, other):
        if isinstance(other, self.__class__):
            return Q_int(mpz(self.q * other.val / self.val), self.q)
        elif (isinstance(other, mpz)):
            return Q_int(mpz(self.q * other / self.val), self.q)
        elif (isinstance(other, int)):
            return Q_int(mpz(self.q * self.q * other / self.val), self.q)
        elif (isinstance(other, float)):
            return Q_int(mpz(self.q * self.q * other / self.val), self.q)
        else:
            raise TypeError(
                "unsupported operand type(s) for // (reverse): '{}' and '{}'".format(self.__class__, type(other)))

    def __truediv__(self, other):
        return self.__floordiv__(other)

    def __rtruediv__(self, other):
        return self.__rfloordiv__(other)

    def __abs__(self):
        return Q_int(np.abs(self.val), self.q)

    def __neg__(self):
        return Q_int(-self.val, self.q)

    def __str__(self):
        return str(self.val/self.q)

    def __format__(self, format_spec):
        return (self.val/self.q).__format__(format_spec)

    @staticmethod
    def convert_matrix(matrix, q_factor):
        (lines, columns) = matrix.shape
        neo_matrix = np.zeros((lines, columns), dtype=object)

        for line in range(lines):
            for column in range(columns):
                val = matrix[line][column]
                neo_matrix[line][column] = Q_int(val*q_factor, q_factor)

        return neo_matrix

    @staticmethod
    def deconvert_matrix(matrix):
        q_factor = matrix[0][0].q
        (lines, columns) = matrix.shape
        neo_matrix = np.zeros((lines, columns))

        for line in range(lines):
            for column in range(columns):
                val = matrix[line][column].val / q_factor
                neo_matrix[line][column] = val

        return neo_matrix
