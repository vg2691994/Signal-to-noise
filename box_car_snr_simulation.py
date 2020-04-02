#!/home/observer/miniconda2/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse, warnings, time
from tqdm import tqdm, trange


def gauss(nsamples, area, fwhm):
  '''
  Generates a gaussian with the given area and fwhm over the given nsamples.
  Nsamples must be odd, so that the peak of the gaussian can be resolved.
  If nsamples is even, one is added to it, with an associated warning.
  '''
  if nsamples%2==0:
    #warnings.warn("Given nsamples ({}) is even. Adding one to it make it \
    #    odd".format(nsamples))
    nsamples += 1
  x = np.arange(-1*nsamples/2 , nsamples/2+1, 1)
  sigma = fwhm / (2*np.sqrt(2*np.log(2)))
  y = 1. / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x+0.25)**2 / (2 * sigma**2) )
  return y*area

def rect_eq_height(nsamples, gauss_area, gauss_fwhm):
  '''
  Generates a rectangular signal with area = gauss_area and height =
  height of a gaussian with fwhm=gauss_fwhm & area = gauss_area.
  Note: The width of the generated rect will be not = gauss_fwhm.
  '''
  #Not required, so not yet implemented

def add_noise(time_series, rms=1, seed=None):
  '''
  Generates gaussian noise of given rms.
  Seed is used if provided.
  Returns (time_series + noise)
  '''
  if seed is not None:
    np.random.seed(seed)

  nsamps = len(time_series)
  noise = np.random.normal(0, rms, nsamps)
  return time_series + noise

def convolve_box_car(time_series, width):
  '''
  Convovles the given time_series with a box-car of given width.
  '''
  return np.convolve(time_series, np.ones(width), mode='valid')

def convolve_box_car_fixed_loc(time_series, width):
  '''
  Convolves the time series with a box-car fixed at the center of 
  the given time series
  '''
  center_of_signal = len(time_series)/2 #len(time_series) is always odd and indexing is from 0
  #Convolution with a box-car will be equal to a simple sum of the 'width' samples around center
  start_of_box_car = center_of_signal - width/2
  end_of_box_car   = center_of_signal + width/2 + (width%2)
  assert end_of_box_car - start_of_box_car == width
  conv_sum = np.sum(time_series[int(start_of_box_car) : int(end_of_box_car)])
  return conv_sum

def maxx_snr_th_eq_height(area, fwhm, rms):
  '''
  Computes the snr of a top-hat signal with area and height
  equal to the area and height of a gaussians signal which has
  the given area and fwhm.
  '''
  sigma_of_gauss = fwhm / (2 * np.sqrt(2 * np.log(2)) )
  height_of_gauss = area / np.sqrt(2 * np.pi * sigma_of_gauss**2)
  width_of_th = area / height_of_gauss
  snr_of_th = area / (rms * np.sqrt(width_of_th))
  return snr_of_th

def maxx_snr_th_eq_width(area, fwhm, rms):
  '''
  Computes the snr of a top-hat signal with area and width
  equal to the area and fwhm of a gaussian signal as provided
  '''
  snr_of_th = area / (rms * np.sqrt(fwhm))
  return snr_of_th


def main():
  area = args.area
  fwhms = np.arange(args.min_w, args.max_w, args.w_sp)
  ntrials = args.ntrials
  rms = 1.

  snrs = np.zeros((len(fwhms), ntrials))
  max_ws = np.empty_like(snrs, dtype=int)
  for ii, fwhm in enumerate(tqdm(fwhms)):
    min_boxcar = 1
    max_boxcar = int(10 * fwhm + 2)
    nsamples = max_boxcar * 5
    boxcars = np.arange(min_boxcar, max_boxcar, 1)

    signal = gauss(nsamples, area, fwhm)
    for trial in range(ntrials):
      time_series = add_noise(signal, rms=rms)
      x_snrs = np.zeros(len(boxcars))
      for jj, boxcar in enumerate(boxcars):
        #Taking the max might be slightly bias inducing
        if args.find_p:
          x_snrs[jj] = np.max(convolve_box_car(time_series, boxcar))
        else:
          x_snrs[jj] = np.max(convolve_box_car_fixed_loc(time_series, boxcar))

      #x_snrs will have that beloved/dreaded curve of snr vs trial_width
      x_snrs = x_snrs / (rms * np.sqrt(boxcars))

      max_ws[ii, trial] = np.argmax(x_snrs)
      snrs[ii, trial] = x_snrs[max_ws[ii, trial]]

  snrs_vs_eq_h = snrs / maxx_snr_th_eq_height(area, fwhms, rms)[:, None]
  snrs_vs_eq_w = snrs / maxx_snr_th_eq_width(area, fwhms, rms)[:, None]

  #plt.plot(fwhms, snrs, '.', ms = 3)
  #plt.plot(fwhms, snrs.mean(axis=1), 'k--', alpha=0.5)
  #plt.plot(fwhms, boxcars[max_ws], '.', ms=3)
  #plt.plot(fwhms, maxx_snr_th_eq_height(area, fwhms, rms), '+')
  plt.errorbar(fwhms, snrs_vs_eq_h.mean(axis=1), yerr = snrs_vs_eq_h.std(axis=1),
      c = 'r', elinewidth = 1, capsize=4, marker='.', linestyle='None', label='Eq_h')

  plt.errorbar(fwhms, snrs_vs_eq_w.mean(axis=1), yerr = snrs_vs_eq_h.std(axis=1),
      c = 'g', elinewidth = 1, capsize=4, marker='.', linestyle='None', label='Eq_w')
  plt.axhline(0.793345, ls="-.", c='k', label="0.793")
  plt.axhline(0.7689, ls="-.", c='k',  label="0.7689")
  plt.legend()
  plt.xlabel("FWHM of the simulated gaussian")
  plt.ylabel("Recovered SNR fraction by a box-car search algo")
  plt.show()


if __name__ == '__main__':
  a = argparse.ArgumentParser()
  a.add_argument("-area", type=float, help="Area of the simulated gaussians (def=100)",
      default=10000.)
  a.add_argument("-min_w", type=float, help="Minimum FWHM to simulate in units of samples\
      (def=0.1)", default=0.1)
  a.add_argument("-max_w", type=float, help="Maximum FWHM to simulate in units of samples\
      (def=10)", default=10.)
  a.add_argument("-w_sp", type=float, help="Spacing of consecutive FWHM sims (def=0.1)",
      default=0.1)
  a.add_argument("-ntrials", type=int, help="Number of trials per sim (def=10)", default=10)
  a.add_argument("-find_p", action='store_true', help="Run a search for peak in\
      box-car convolution, instead of convolving just at the right location (def:False)",\
      default=False)

  args = a.parse_args()
  main()
















  

