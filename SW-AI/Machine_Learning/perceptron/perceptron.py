import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
	def __init__(self, eta=0.01, n_iter=10, early_stop=3):
		self.eta = eta
		self.n_iter = n_iter
		self.early_stop = early_stop

	def fit(self, X, y):
		self.w_ = np.zeros((1 + X.shape[1],))
		self.errors_ = []

		for k in range(self.n_iter):
			errors = 0
			for idx in range(len(X)):
				err = y[idx] - self.predict(X[idx]) # 답을 맞춘경우 0 이 되어 가중치를 업데이트하지 않는다.
				self.w_[1:] += (self.eta * err) * X[idx]
				self.w_[0] += self.eta * err
				if err != 0:
					errors += 1
			self.errors_.append(errors)
			# # self.early_stop 횟수 만큼 err의 변화가 없다면 조기 종료
			# if len(self.errors_) >= self.early_stop: # early stop
			# 	if len(set(self.errors_[-self.early_stop:])) == 1:
			# 		print('early stop! {} iters, \ntrain_accuracy : {}'.format(k+1,1 - self.errors_[-1]/len(X)))
			# 		break
		return self

	def net_input(self, X):
		return (X @ self.w_[1:]) + self.w_[0]

	def predict(self, X):
		return np.where(self.net_input(X) >= 0., 1, -1)

class ovr_perceptron():
	def __init__(self, models):
		self.models = models

	def fit(self, X, y):
		classes = np.unique(y)
		if len(self.models) != len(classes):
			print('Number of model and class do not match.')
			return 0
		for idx, class_ in enumerate(classes):
			tmp_y = np.where(y==class_,1,-1)
			self.models[idx].fit(X, tmp_y)
		return self

	def predict(self,X):
		out = self.models[0].predict(X)
		for model in self.models[1:]:
			out = np.vstack((out,model.predict(X)))
		out = np.argmax(out.T,axis=1)
		print(out)
		return out

class my_mlp(): # hidden_layer = 1

	def __init__(self, eta=0.01, n_iter=10, hidden_unit=5, early_stop = 5, batch_size=4):
		self.eta = eta
		self.n_iter = n_iter
		self.hidden_unit = hidden_unit
		self.early_stop = early_stop
		self.batch = batch_size

	def to_one_hot(self, y):
		n_class = np.amax(y) + 1
		bin = np.zeros((len(y), n_class))
		for i in range(len(y)):
			bin[i, y[i]] = 1.
		return bin

	def sigmoid(self, X):
		return 1 / (1 + np.exp(-X))

	def sig_grad(self, X):
		return self.sigmoid(X) * (1 - self.sigmoid(X))

	def fit(self, X, y):
		self.classes = np.unique(y)
		# with bias
		self.w_1 = np.random.random((1 + X.shape[1], self.hidden_unit))
		self.w_2 = np.random.random((1 + self.hidden_unit, len(self.classes)))
		# # no bias
		# self.w_1 = np.random.random((X.shape[1], self.hidden_unit))
		# self.w_2 = np.random.random((self.hidden_unit, len(self.classes)))

		self.errors_ = []
		self.y = self.to_one_hot(y)
		for k in range(self.n_iter):
			errors = 0
			for idx in range(0,len(X),self.batch):
				# Forward
				out_1 = self.sigmoid( (X[idx:idx + self.batch] @ self.w_1[1:,:]) + self.w_1[0,:])
				out_2 = self.sigmoid(out_1 @ self.w_2[1:,:] + self.w_2[0,:])

				# get gradient
				err_2 = self.y[idx:idx + self.batch] - out_2
				delta_2 = err_2 * self.sig_grad(out_2)

				err_1 = delta_2 @ self.w_2.T
				delta_1 = err_1[:,1:] * self.sig_grad(out_1)

				# back propagation
				self.w_2[1:,] += out_1[1:,].T @ delta_2[1:] * self.eta # w2
				self.w_2[0,:] += err_2[0,:] * self.eta # b2
				self.w_1[1:,] += X[idx:idx + self.batch].T @ delta_1 * self.eta # w1
				self.w_1[0,:] += err_1[0,1:] * self.eta # b1

				# # Forward (no bias)
				# out_1 = self.sigmoid(X[idx:idx + self.batch] @ self.w_1)
				# out_2 = self.sigmoid(out_1 @ self.w_2)
				#
				# # get gradient
				# err_2 = self.y[idx:idx + self.batch] - out_2
				# delta_2 = err_2 * self.sig_grad(out_2)
				#
				# err_1 = delta_2 @ self.w_2.T
				# delta_1 = err_1* self.sig_grad(out_1)
				#
				# # back propagation
				# self.w_2 += out_1.T @ delta_2 * self.eta  # w2 ()
				# self.w_1 += X[idx:idx + self.batch].T @ delta_1 * self.eta

				if np.sum(err_2) != 0:
					errors += 1
			self.errors_.append(errors)

			# # early stop
			# if len(self.errors_) >= self.early_stop: # early stop
			# 	if len(set(self.errors_[-self.early_stop:])) == 1:
			# 		print('early stop! {} iters, \ntrain_accuracy : {}'.format(k+1,1 - self.errors_[-1]/len(X)))
			# 		break
		return self

	def net_input(self, X):
		# with bias
		out = self.sigmoid((X @ self.w_1[1:, :]) + self.w_1[0, :])
		out = self.sigmoid(out @ self.w_2[1:, :] + self.w_2[0, :])
		# # no bias
		# out = self.sigmoid(X @ self.w_1)
		# out = self.sigmoid(out @ self.w_2)
		return out

	def predict(self, X):
		return np.argmax(self.net_input(X),axis=1)  # one-hot label to label [0,1,0] -> 1