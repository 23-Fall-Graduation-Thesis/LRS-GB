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


# only GB - score
class LRSGBCondition(ConditionBase):
    def __init__(self):
        pass

    def init(self, thr_init_score):
        super().__init__()
        self.thr_init_score = thr_init_score

    def check_condition(self, init_weva_try, target_init_weva_set) :
        check_GB = True
        if len(target_init_weva_set) > 0:
            target_init_weva = target_init_weva_set[-1]
            init_score = self.get_init_score(init_weva_try[:-1], target_init_weva) # LRS_score
        else:
            raise ValueError("target init weva must be calculated before check condition")

        if init_score < self.thr_init_score:
            check_GB = False
        
        return check_GB, init_score
    
    
    def adjust_condition(self):
        pass
    
    
    def get_condition(self):
        return self.thr_init_score
    
    
    def get_init_score(self, init_weva, target_init_weva):
        score = 0.
        for cur, target in zip(init_weva, target_init_weva):
            err = max(0, cur-target)
            score += 1/(1+err)
        score /= len(init_weva)
        return score

# new score of GB weva
class GBwevaCondition(ConditionBase):
    def __init__(self):
        pass

    def init(self, thr_init_score, lamb=5):
        super().__init__()
        self.thr_init_score = thr_init_score
        self.lamb = lamb

    def check_condition(self, weva_try, target_weva_set) :
        check_GB = True
        if len(target_weva_set) > 0:
            target_weva = target_weva_set[-1]
            init_score = self.get_init_score(weva_try[:-1], target_weva) # LRS_score
        else:
            raise ValueError("target init weva must be calculated before check condition")

        if init_score < self.thr_init_score:
            check_GB = False
        
        return check_GB, init_score
    
    
    def adjust_condition(self):
        pass
    
    
    def get_condition(self):
        return self.thr_init_score
    
    def get_layer_score(self):
        return self.score_list
    
    def get_init_score(self, init_weva, target_init_weva):
        self.score_list = []
        score = 1.
        for cur, target in zip(init_weva, target_init_weva):
            if cur < target:
                err = (target-cur)/target
            else:
                err = min(self.lamb * (cur-target)/target, 1)
            self.score_list.append(1-err >= self.thr_init_score) # [True, True, False, ..]
            score = min(score, 1 - err)
        return score
    
    
# new score of GB weva (using target weva's sum)
class GBwevaSumCondition(ConditionBase):
    def __init__(self):
        pass


    def init(self, thr_init_score, lamb=5):
        super().__init__()
        self.thr_init_score = thr_init_score
        self.lamb = lamb
        self.flag = True
        self.cumulated_weva = []
        self.cumulated_target_weva = []


    def check_condition(self, weva_try, target_weva_set) :
        check_GB = True
        if len(target_weva_set) > 0:
            target_weva = target_weva_set[-1]
            init_score = self.get_init_score(weva_try[:-1], target_weva) # LRS_score
        else:
            raise ValueError("target init weva must be calculated before check condition")

        if init_score < self.thr_init_score:
            check_GB = False
        
        return check_GB, init_score, target_weva
    
    
    def adjust_condition(self):
        pass
    
    
    def get_condition(self):
        return self.thr_init_score
    
    
    def get_layer_score(self):
        return self.score_list
    
    
    def get_init_score(self, weva, target_weva):
        self.score_list = []
        score_list = []
        score = 1.
        
        if len(self.cumulated_target_weva) == 0:
            for cur, target in zip(weva, target_weva):
                if cur < target:
                    err = (target-cur)/target
                else:
                    err = min(self.lamb * (cur-target)/target, 1)
                self.score_list.append(1-err >= self.thr_init_score) # [True, True, False, ..]
                score_list.append(1-err)
                score = min(score, 1 - err)
                # print(score)
            
        else:  
            for cum_cur, cur, cum_target, target in zip(self.cumulated_weva, weva, self.cumulated_target_weva, target_weva):
                tmp_cur, tmp_target = cum_cur+cur, cum_target+target
                if tmp_cur < tmp_target:
                    err = (tmp_target-tmp_cur)/tmp_target
                else:
                    err = min(self.lamb * (tmp_cur-tmp_target)/tmp_target, 1)
                self.score_list.append(1-err >= self.thr_init_score) # [True, True, False, ..]
                score_list.append(1-err)
                score = min(score, 1 - err)
                # print(score)
        
        # print(score, sum(score_list)/len(score_list))
        mean_score = sum(score_list)/len(score_list)
        # return score
        return mean_score


    def adjust_bound(self, weva, target_weva):
        for i, (cur, target) in enumerate(zip(weva, target_weva)):
            if self.flag:
                self.cumulated_weva.append(cur)
                self.cumulated_target_weva.append(target)
            else:
                self.cumulated_weva[i] += cur
                self.cumulated_target_weva[i] += target
        
        if self.flag: self.flag = False