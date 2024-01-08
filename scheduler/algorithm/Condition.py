from abc import ABC, abstractmethod

class ConditionBase(ABC):
    def __init__(self):
        pass

    def sigma_function(self, weva):
        # weva = weva[1:-self.mlast]
        weva_index = [weva.index(x) for x in sorted(weva)]
        return weva_index
        
    @abstractmethod
    def check_condition(self):
        """조건을 만족하는지 확인합니다.
        Returns:
            bool: 만족하면 True, 만족하지 않으면 False를 반환합니다.
        """
        pass
    
    @abstractmethod
    def adjust_condition(self):
        pass
        
    @abstractmethod
    def get_condition(self):
        pass


class AutoLRCondition(ConditionBase):
    def __init__(self, thr_score):
        super().__init__()
        self.thr_score = thr_score


    def check_condition(self, weva_try) :
        weva_idx = self.sigma_function(weva_try[:-1])
        score = self.get_score(weva_idx)
        if score >= self.thr_score:
            return True
        else:
            return False


    def get_condition(self):
        return self.thr_score


    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
            
        return 1.0 - diff / len(A) ** 2 * 2

# Trial1
class LRSGBCondition(ConditionBase):
    def __init__(self, thr_score, constraints):
        super().__init__()
        self.thr_score = thr_score
        self.constraints = constraints


    def check_condition(self, weva_try, init_weva_try) :
        check_autoLR, check_GB = True, True
        weva_idx = self.sigma_function(weva_try[:-1])
        score = self.get_score(weva_idx)
        for i in range(len(init_weva_try)):
            if init_weva_try[i] > self.constraints[i]:
                check_GB = False
                break    

        if score < self.thr_score:
            check_autoLR = False
        
        return check_autoLR, check_GB
    
    
    def get_condition(self):
        return self.thr_score, self.constraints
    
    
    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
        
        return 1.0 - diff / len(A) ** 2 * 2