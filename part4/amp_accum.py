import torch

class AmpAccum:
    def __init__(self,optimiser,accum_steps:int=4,amp:bool=True):
        self.optimiser=optimiser
        self.accum_steps=accum_steps
        self._n=0
        self.amp=torch.cuda.is_available() and amp

        self.scaler=torch.amp.GradScaler(enabled=self.amp)

    def backward(self,loss:torch.Tensor):
        loss=loss/self.accum_steps
        if self.amp:
            self.scaler.scale(loss).backward()

        else:
            loss.backward()
        self._n+=1
    
    def should_step(self)->bool:
        return (self._n%self.accum_steps==0)
    
    def step(self):
        if self.amp:
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            self.optimiser.step()

    def zero_grad(self):
        self.optimiser.zero_grad(set_to_none=True)