# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:17:40 2022

@author: Administrator
"""

import random
import math

class UniformSampling:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def get(self):
        return random.uniform(self.min, self.max)

class RoundedSampling:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def get(self):
        min = int(self.min)
        max = int(self.max)
        range = max - min
        
        tentative_step_size = self.__round_up_to_power_of_ten(range)
        
        # i.e. 10, 100, 1000
        power_of_ten = self.__get_step_size(tentative_step_size, range)

        num_digits_power_of_ten = self.__number_of_digits(power_of_ten)

        start = min
        if (power_of_ten > 1):
            start = self.__get_starting_value(min, num_digits_power_of_ten)
        
        result = random.randrange(start, max + 1, power_of_ten)
        return float(result)

    
    def __get_step_size(self, step_size, range):
        slots = range // step_size
        if (slots >= 7 and slots <= 30):
            return step_size
        if (slots < 7):
            new_step_size = step_size // 10
            if new_step_size == 0:
                return new_step_size
            return self.__get_step_size(new_step_size, range)
        return self.__get_step_size(step_size * 10, range)


    def __round_up_to_power_of_ten(self, num):
        exp = math.ceil(math.log10(num))
        return 10 ** exp


    def __number_of_digits(self, num):
        digits = int(math.log10(num)) + 1
        return digits

    
    def __get_starting_value(self, min, num_digits_stepsize):
        value = round(min, num_digits_stepsize * -1)
        if value < min:
            value += 10 ** (num_digits_stepsize - 1)
        return value
    
def split_sampling(from_set,split):
    from_list = list(from_set)
    random.shuffle(from_list)
    result = [from_list[sum(split[:i]):sum(split[:(i+1)])] for i in range(len(split))]
    return result
            
def day_sampling(start_day,start_range,end_day,end_range,dist="uniform"):
    if dist=="uniform":
        if start_day is not None and start_range is not None:
            start = start_day + random.randrange(start_range)
        else:
            start = -1
        if end_day is not None and end_range is not None:
            end = end_day - random.randrange(end_range)
        else:
            end = -1
    return start,end
    
class RandomAmount:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def getAmount(self):
        return random.uniform(self.min, self.max)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    