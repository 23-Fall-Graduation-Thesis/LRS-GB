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
    def __init__(self):
        pass
        

    def init(self, thr_score):
        super().__init__()
        self.thr_score = thr_score


    def check_condition(self, weva_try) :
        weva_idx = self.sigma_function(weva_try[:-1])
        score = round(self.get_score(weva_idx), 3)
        if score >= self.thr_score:
            return True, score
        else:
            return False, score


    def get_condition(self):
        return self.thr_score

    
    def adjust_condition(self):
        pass
    

    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
            
        return 1.0 - diff / len(A) ** 2 * 2

# Trial1
class LRSGBCondition(ConditionBase):
    def __init__(self):
        pass
    
    
    def init(self, thr_score, constraints):
        super().__init__()
        self.thr_score = thr_score
        self.constraints = constraints


    def check_condition(self, weva_try, init_weva_try) :
        check_autoLR, check_GB = True, True
        weva_idx = self.sigma_function(weva_try[:-1])
        score = self.get_score(weva_idx)
        for i in range(len(self.constraints)):
            if init_weva_try[i] > self.constraints[i]:
                check_GB = False
                break    

        if score < self.thr_score:
            check_autoLR = False
        
        return check_autoLR, check_GB, score
    
    
    def adjust_condition(self):
        pass
    
    
    def get_condition(self):
        return self.thr_score, self.constraints
    
    
    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
        
        return 1.0 - diff / len(A) ** 2 * 2
    
# Trial2
class LRSScoreCondition(ConditionBase):
    def __init__(self):
        pass
    
    
    def init(self, thr_score, thr_init_score):
        super().__init__()
        self.thr_score = thr_score
        self.thr_init_score = thr_init_score


    def check_condition(self, weva_try, init_weva_try, init_weva_target) :
        check_autoLR, check_GB = True, True
        weva_idx = self.sigma_function(weva_try[:-1])
        score = self.get_score(weva_idx) # AutoLR score
        init_score = self.get_init_score(init_weva_try[:-1], init_weva_target[:-1]) # LRS_score

        if score < self.thr_score:
            check_autoLR = False
        if init_score < self.thr_init_score:
            check_GB = False
        
        return check_autoLR, check_GB, score
    
    
    def adjust_condition(self):
        pass
    
    
    def get_condition(self):
        return self.thr_score, self.constraints
    
    
    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
        
        return 1.0 - diff / len(A) ** 2 * 2
    
    def get_init_score(self, init_weva, target_init_weva):
        score = 0.
        for cur, target in zip(init_weva, target_init_weva):
            err = max(0, cur-target)
            score += 1/(1+err)
        score /= len(init_weva)
        return score