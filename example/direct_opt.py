import os
import sys
cwd = os.getcwd()

sys.path.insert(0,os.path.join(cwd, ".."))
import torch
from torchddsp import losses, synths, core
import tqdm

torch.set_default_device("cuda")
n_sources = 2
n_frames = 250

class DirectOptimization(torch.nn.Module):
    def __init__(self, estimate_f0=True):
      super().__init__()
      self.source_filter_synth = synths.SourceFilterSynth2(
            n_samples=64000,
            sample_rate=16000,
            n_harmonics=101,
            audio_frame_size=512,
            hp_cutoff=500,
            f_ref=torch.tensor(200, dtype=torch.float32),
            estimate_voiced_noise_mag=True,
      )
      self.harmonic_amp_params = torch.nn.Parameter(torch.ones(n_sources, n_frames, 1))
      self.harmonic_roll_off_params = torch.nn.Parameter(torch.ones(n_sources, n_frames, 1))
      self.noise_gain_params = torch.nn.Parameter(torch.ones(n_sources, n_frames, 1))
      self.lin_spec_params = torch.nn.Parameter(torch.ones(n_sources, n_frames, 21))
      self.noise_mag_params = torch.nn.Parameter(torch.ones(n_sources, 1, 40))
      self.estimate_f0 = estimate_f0
      if estimate_f0:
        self.f0_p = torch.nn.Parameter(torch.rand(1, n_frames, n_sources))

    def forward(self, f0_hz=None):
      batch_size = 1

      if self.estimate_f0:
        f0_hz = self.f0_p
      f0_hz = f0_hz.transpose(1, 2)  # [batch_size, n_sources, n_freq_frames]
      f0_hz = torch.reshape(
          f0_hz, (batch_size * n_sources, -1)
      )  # [batch_size * n_sources, n_freq_frames]
      f0_hz = core.resample(f0_hz, n_frames)

      voiced_unvoiced = torch.where(
          f0_hz > 0.0,
          torch.tensor(1.0, device=f0_hz.device),
          torch.tensor(0.0, device=f0_hz.device),
      )[:, :, None]
      f0_hz = f0_hz[:, :, None]  # [batch_size * n_sources, n_frames, 1]
      signal = self.source_filter_synth(
              self.harmonic_amp_params,
              self.harmonic_roll_off_params,
              f0_hz,
              self.noise_gain_params,
              voiced_unvoiced,
              self.lin_spec_params,
              self.noise_mag_params,
      )
      sources = torch.reshape(signal, (1, n_sources, -1))
      mix = torch.sum(sources, dim=1)
      
      return mix, sources


model_direct_opt = DirectOptimization()
out, sources = model_direct_opt()
print(f"out shape: {out.shape} {sources.shape}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


d_model = DirectOptimization()

optimizer = torch.optim.Adam(d_model.parameters(), lr=0.01, weight_decay=0.00001)
loss_container = AverageMeter()
d_model.train()

data = torch.load("./data.pt")

pbar = tqdm.tqdm(range(10000))
for i in pbar:
  if (i % 100) == 0:
      out, sources = d_model()
      from IPython.display import display
      print("remix:")
      print(f"sources shape: {sources.shape}")
      for i, source in enumerate(sources[0]):
        print(f"source {i}:")

  x = data[0].unsqueeze(0)  # mix
  x = x.to("cuda")
  optimizer.zero_grad()
  y_hat, _ = d_model()
  loss_fn = losses.SpectralLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64],
        mag_weight=1,
        logmag_weight=1,
        logmel_weight=0,
        delta_freq_weight=0,
        delta_time_weight=0,
  )
  loss = loss_fn(x, y_hat)
  loss.backward()
  optimizer.step()
  loss_container.update(loss.item())
  pbar.set_postfix({ "loss": loss_container.avg })