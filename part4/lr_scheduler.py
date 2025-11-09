import math

class LRScheduler:
    def __init__(self,optimiser,base_lr,total_steps,warmup_steps):
        self.optimiser=optimiser
        self.base_lr=base_lr
        self.total_steps=total_steps
        self.warmup_steps=warmup_steps
        self.current_step=0

    def step(self):
        self.current_step+=1
        if self.current_step<self.warmup_steps:
            lr=self.base_lr*(self.current_step/self.warmup_steps)
        else:
            prog=(self.current_step-self.warmup_steps)/(self.total_steps-self.warmup_steps)
            lr=self.base_lr*0.5*(1+math.cos(math.pi*prog))
        for g in self.optimiser.param_groups:
            g['lr']=lr
        return lr