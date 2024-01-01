# DoubleLinkedList
from multipledispatch import dispatch
from abc import ABCMeta, abstractproperty

class Node:
    __metaclass__ = ABCMeta

    @abstractproperty
    def pref():
        """Ссылка на предыдущий объект"""

    @abstractproperty
    def nref():
        """Ссылка на следующий объект"""
        
class DoubleLinkedList:
    def __init__(self) -> None:
        self.head = None
        self.tail = None
    
    def _insert_start(self, obj: Node) -> bool:
        if not self.head:
            self.head = obj
            self.tail = self.head
            return True
        return False

    @dispatch(object)
    def insert(self, obj: Node):
        if self._insert_start(obj):
            return self

        obj.pref = self.tail
        self.tail.nref = obj
        
        self.tail = obj
        return self

    @dispatch(list)
    def insert(self, obj: list):
        while obj:
            self.insert(obj[0])
            obj.pop(0)
        return self


from enum import Enum
import numpy as np

class Func(Enum):
    linear  = (lambda x: x, 
               lambda x: 1)
    
    sigmoid = (lambda x: np.nan_to_num(1 / (1 + np.exp(-x))), 
               lambda x: np.nan_to_num(np.exp(x) / (1 + np.exp(x))**2))
    
    relu    = (lambda x: np.maximum(0, x),
               lambda x: (x > 0) * 1)
    
    tanh    = (lambda x: np.nan_to_num((np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)),
               lambda x: np.nan_to_num(4 * np.exp(2 * x) / (np.exp(2 * x) + 1)**2))
    
    mse     = (lambda y_true, y_pred: (y_true - y_pred)**2,
               lambda y_true, y_pred: 2 * (y_true - y_pred))

mul = np.dot
def null(x) -> bool: return x.shape[0] == 0 if type(x) is np.ndarray else not bool(x)

class Layer(Node):
    def __init__(self, n_neurons: int, actFunc: Func, lmbd: float = 0.1) -> None:
        self.n_neurons = n_neurons
        self.actFunc, self.actFuncDer = actFunc.value
        self.lmbd = lmbd
        self.W, self.B = None, None
        self.T, self.H = None, None
        self._pref = None
        self._nref = None

    def update_weights(self) -> None:
        if null(self.W) and null(self.B):
            self.W = np.random.uniform(-.5, .5, (self.n_neurons,self.nref.n_neurons))
            self.B = np.random.uniform(-.5, .5, (1, self.nref.n_neurons))
            
            self.dE_dW = np.zeros(self.W.shape)
            self.dE_dB = np.zeros(self.B.shape)
        else:
            self.W = self.W + self.lmbd * self.dE_dW
            self.B = self.B + self.lmbd * self.dE_dB

    def transform(self, X: np.ndarray) -> np.ndarray:
        # если мы в последнем слое, то применяем функцию активации
        if not self.nref:
            return self.actFunc(X)

        # обновление весов
        self.update_weights()

        # если это первый слой, инициализируем h
        self.H = X if not self.pref else self.H

        # считаем сумму T и применяем функцию активации для
        # нахождения H для следующего слоя
        self.nref.T = mul(self.H, self.W) + self.B
        self.nref.H = self.actFunc(self.nref.T)
        # возвращаем вектор для следующего слоя
        return self.nref.H

    def backprop(self, dE_dH):
        # если это последний слой, принимаем производную ошибки
        # иначе считаем как dE_dT_(i + 1) * W_T
        self.dE_dH = dE_dH if not self.nref else mul(self.nref.dE_dT, self.W.T)
        self.dE_dT = self.dE_dH * self.actFuncDer(self.T)
        
        # градиент весов предыдущего слоя
        self.pref.dE_dW = mul(self.pref.H.T, self.dE_dT)
        
        # градиент смещения предыдущего слоя
        self.pref.dE_dB = self.dE_dT

    @property
    def pref(self): return self._pref

    @pref.setter
    def pref(self, obj): self._pref = obj
    
    @property
    def nref(self): return self._nref

    @nref.setter
    def nref(self, obj): self._nref = obj
	
class NeuralNetwork:
	def __init__(self, layers: list[Layer], lossFunc: Func) -> None:
		# двусвязный список s1 <--> s2 <--> s3 <--> s4
		self.layers = DoubleLinkedList().insert(layers)
		# функция потерь и её производная
		self.lossFunc, self.lossFuncDer = lossFunc.value

	def fit(self, X: np.ndarray, Y: np.ndarray, n_epohs: int = 400, eps: float = 0.0001):
		self.answer = np.zeros((X.shape[0],))
		for _ in range(n_epohs):
			last_answer = np.zeros((X.shape[0],))
			for i in range(X.shape[0]):
				x = X[i].reshape(1, X[i].shape[0])
			
				# прямое распространение
				layer = self.layers.head
				vector = x
				while layer != None:
					vector = layer.transform(vector)
					layer = layer.nref
				
				#градиент ошибки
				dE_dH = self.lossFuncDer(Y[i], vector)
				
				# обратное распространение
				layer = self.layers.tail
				while layer.pref != None:
					layer.backprop(dE_dH)
					layer = layer.pref
				last_answer[i] = vector
			
			# точка остановки градиентного спуска
			if (np.fabs(self.answer - last_answer) < eps).all():
				break
			self.answer = last_answer
		return self
	
	def predict(self, X: np.ndarray):
		answer = np.empty((0,self.layers.tail.n_neurons))
		for i in range(X.shape[0]):
			x = X[i].reshape(1, X[i].shape[0])
			
			# прямое распространение
			layer = self.layers.head
			vector = x
			while layer != None:
				vector = layer.transform(vector)
				layer = layer.nref
			answer = np.vstack((answer,vector))
		return answer
