import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        '''
        연산 함수 저장(연결 기록)
        '''
        self.creator = func
        
    def backward_recur(self):
        f = self.creator # 어떤 함수사용했는지 가져오기
        if f is not None:
            x = f.input # 입력 가져오기
            x.grad = f.backward(self.grad) # 역전파 계산
            print(x.data,x.grad)
            x.backward_recur() # 재귀로 호출
    
    def backward(self):
        funcs = [self.creator]
        while(funcs):
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            print(x.data,x.grad)
            if x.creator is not None:
                funcs.append(x.creator)
        
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        
        output = Variable(y)
        output.set_creator(self)
        
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        '''
        연산 작성
        '''
        raise NotImplementedError() # 구현 안되어 있음을 의미
        
    def backward(self, gy):
        '''
        gy는 chain rule로 곱해주기 위해 이전의 기울기
        '''
        raise NotImplementedError()
        
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 *  x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
    

    
def numerical_diff(f, x, eps=1e-4):
    '''
    numerical diff
    수치 미분으로 f(x + h) - f(x - h) / 2*h : 
                  h : lim -> 0
    '''
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def f(x):
    '''
    composite function diff
    '''
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))