'''소프트웨어 꼰대강의
Practice for cross entropy loss
'''

from typing import List
import numpy as np
import matplotlib.pyplot as plt

class BCELoss():
    '''Class: Binary Cross Entropy Loss'''
    def __init__(self, label: bool = True) -> None:
        self.label = label
        self.p = np.linspace(start=0.001, stop=1.0, num=1000)
    
    def get_cross_entropy(self) -> list:
        '''Random var (Label: Treu, False) 값이 주어졌을 경우 엔트로피 분포 리턴
            Entropy = -p * log(p) -> -log(p)
                        -1*log(1-p)
            정답(label) True, False 모든 경우에 1을 곱해주게 됩니다.
        '''
        return -1*np.log2(self.p) if self.label else -1*np.log2(1-self.p)
    
    def get_squared_error(self) -> list:
        '''제곱 오차 확률 분포 리턴
        (y - y_hat)^2
        '''
        if self.label:
            squared_error = 1 + (-2*self.p) + np.square(self.p)
        else:
            squared_error = 1 + (-2*(1-self.p)) + np.square((1-self.p))
        return squared_error
    
    def plot_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = None,
        y_label: str = None,
    ) -> None:
        '''x, y 값을 받아서 그래프 생성'''
        plt.xlabel('Probability')
        y_label = y_label if y_label else 'Result'
        plt.ylabel(y_label)
        title = title if title else 'no-title'
        plt.title(title)
        plt.plot(x, y)
        plt.savefig(f'{title}.png', dpi=300)
        plt.show()
        
    def combine_plots(
        self,
        ys: List[tuple],
        title: str = None,
        y_label: str = None
    ) -> None:
        '''여러개 그림(데이터) 들어올 경우 합쳐서 하나의 평면에 그리기
        Ex. ys = [('True', array), ('False', array), ...]
        '''
        plt.xlabel('Probability')
        y_label = y_label if y_label else '-plog(p)'
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        for  label, y in ys:
            plt.plot(self.p, y, label=label)
        plt.legend()
        plt.savefig(f'{title}.png', dpi=300)
        plt.show()
               
        
def main() -> None:
    '''main 함수'''
    
    # Label True/False 인 경우의 처리
    for label in [True, False]:
        bce = BCELoss(label)
        bce.plot_data(
            x=bce.p,
            y=bce.get_cross_entropy(),
            title=f'Label_{bce.label}, log(p)',
            y_label='-plog(p)'
        )
        
    # Cross entropy 합쳐진 plot 그리기
    ys_entropy = [
        (True, BCELoss(True).get_cross_entropy()),
        (False, BCELoss(False).get_cross_entropy()),
    ]
    BCELoss().combine_plots(ys=ys_entropy, title='BCE_combined')
    
    # True인 경우 Squared error와 Entropy 비교 plot
    ys_squared_error = [
        ('entropy (True)', BCELoss(True).get_cross_entropy()),
        ('Squared Error (True)', BCELoss(True).get_squared_error()),
    ]
    BCELoss().combine_plots(ys=ys_squared_error, title='BCE+SE_combined_True', y_label='-plog(p)/SE')
    
    # False 경우 Squared error와 Entropy 비교 plot
    ys_squared_error = [
        ('entropy (False)', BCELoss(False).get_cross_entropy()),
        ('Squared Error (False)', BCELoss(False).get_squared_error()),
    ]
    BCELoss().combine_plots(ys=ys_squared_error, title='BCE+SE_combined_False', y_label='-plog(p)/SE')
    
    # 모든 Squared error와 Entropy 비교 plot
    ys_squared_error = [
        ('entropy (True)', BCELoss(True).get_cross_entropy()),
        ('Squared Error (True)', BCELoss(True).get_squared_error()),
        ('entropy (False)', BCELoss(False).get_cross_entropy()),
        ('Squared Error (False)', BCELoss(False).get_squared_error()),
    ]
    BCELoss().combine_plots(ys=ys_squared_error, title='BCE+SE_combined_all', y_label='-plog(p)/SE')
    
    
if __name__=='__main__':
    main()
    # bce = BCELoss(True)
    # ce = bce.get_cross_entropy()
    # se = bce.get_squared_error()
    # bce.plot_data(bce.p, se, title='bce_true', y_label='-plog(p)')
    # print(se)
    