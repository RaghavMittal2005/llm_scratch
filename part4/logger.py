from __future__ import annotations
import torch
import time
from pathlib import Path

class Nooplogger:
    """A logger in case of no logging"""
    def close(self):
        pass
    def log(self,**kwargs):
        pass

class TensorboardLogger:
    """
    Backward compatible:
      - logger.log(step=..., loss=..., lr=...)
    Extras you can optionally use:
      - logger.hist("params/wte.weight", tensor, step)
      - logger.text("samples/generation", text, step)
      - logger.image("attn/heatmap", HWC_or_CHW_tensor_or_np, step)
      - logger.graph(model, example_batch)
      - logger.hparams(dict_of_config, dict_of_metrics_once)
      - logger.flush()
    Auto-behavior:
      - If a value in .log(...) is a tensor/ndarray with >1 element, it logs a histogram.
      - If key starts with "text/", logs as text.
    """
    # logger.py
    def __init__(self, out_dir: str, flush_secs: int = 10, run_name: str | None = None):
        self.w=None
        run_name=time.strftime("%Y%m%d-%H%M%S") if run_name is None else run_name
        self.h_params_logged=False
        run_dir=Path(out_dir)/run_name
        run_dir.mkdir(parents=True,exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.w=SummaryWriter(log_dir=str(run_dir),flush_secs=flush_secs)
        except Exception as e :
            print(f"[TBLogger] TensorBoard not available: {e}. Logging disabled.")
            self.auto_max_his_elem=2048
            self.run_dir=Path(run_dir)

    def log(self,step:int|None,**kv):
        if self.w is None:return
        for k ,v in kv.items():
            if isinstance(k,str) or k.startswith('text/'):
                self.w.add_text(k[5:],global_step=step)
            
            try:
                import numpy as np
                istorch=isinstance(v,torch.Tensor)
                isnumer=isinstance(v,np.ndarray)
                if isnumer or istorch:
                    numel=int(v.size if isnumer else v.numel())
                    if numel==1:
                        self.w.add_scalar(k,v,global_step=step)
                    else:
                        if numel<=self.auto_max_his_elem:
                           self.w.add_histogram(k,v.detach().cpu() if istorch else v,global_step=step)
                        else:
                           arr=v.detach().cpu().flatten().numpy() if istorch else v.flatten()
                           self.w.add_scalar(k+"/mean",float(arr.mean()),global_step=step)
                           self.w.add_scalar(k+"/std",float(arr.std()),global_step=step)

                continue
            except Exception as e:
                pass
            
            try:
                self.w.add_scalar(k,float(v),global_step=step)
            except Exception as e:
                pass
    def hist(self, tag: str, values: any, step: int |None, bins: str = "tensorflow"):
        if self.w is None:return
        try:
            if isinstance(values,torch.Tensor):
                values=values.detach().cpu()
            self.w.add_histogram(tag,values,step,bins)

        except Exception:
            pass

    def text(self,tag:str,text:str,step=int|None):
        if self.w is None:return
        try:
            self.w.add_text(tag,text,step)
        except Exception:
            pass
    def image(self, tag: str, img, step: int|None):
        """
        img: torch.Tensor [C,H,W] or [H,W,C] or numpy array
        """
        if not self.w: return
        try:
            self.w.add_image(tag, img, global_step=step, dataformats="CHW" if getattr(img, "ndim", 0) == 3 and img.shape[0] in (1,3) else "HWC")
        except Exception:
            pass

    def graph(self, model, example_input):
        if not self.w: return
        try:
            # example_input: a Tensor batch or a tuple
            if not isinstance(example_input, tuple):
                example_input = (example_input,)
            self.w.add_graph(model, example_input)
        except Exception:
            pass  # graph tracing can fail depending on model control flow; don't crash

    def hparams(self, hparams: dict[str, any], metrics_once: dict[str, float]|None=None):
        if not self.w or self.hparams_logged:
            return
        try:
            # Single, stable sub-run so it doesn’t spam the left pane
            self.w.add_hparams(hparams, metrics_once or {}, run_name="_hparams")
            self.hparams_logged = True
        except Exception:
            pass

    def flush(self):
        if self.w:
            try: self.w.flush()
            except Exception: pass

    def close(self):
        if self.w:
            try: self.w.close()
            except Exception: pass


class WBLogger:
    def __init__(self,project,run_name:str|None=None):

        try:
            import wandb
            wandb.init(project=project,name=run_name)
            self.wb=wandb
        except Exception:
            print("wandb error")
    def log(self,**kv):
        if self.wb:self.wb.log(kv)       

def init_logger(which: str, out_dir: str = "runs/part4"):
    if which == 'tensorboard':
        tb = TensorboardLogger(out_dir)
        return tb if tb.w is not None else Nooplogger()
    if which == 'wandb':
        return WBLogger(project='llm-part4')
    return Nooplogger()

