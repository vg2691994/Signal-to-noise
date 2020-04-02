#!/home/observer/miniconda2/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

tsamp = 1 #seconds
rms_noise = 1.0

#No. of box-car trials 
N_bc_trials = 100

#Defining an array of all box-car trials
#Smallest box-car is 1 sample wide
#Largest box-car is N_bc_trials samples wide
#i.e. each box-car trial is one sample wider than the previous
bc_trials = np.arange(tsamp, N_bc_trials * tsamp, tsamp)


#Let us take our signal to be the normal function
#f(x) = 1 / sqrt(2 pi sigma**2) * exp ( - x**2 / (2 sigma**2) ) 


#The area under the f(x) within w samples from center is given by the erf
def area_under_w_samples(w, sigma, Tot_area):
  return erf(w / (2. * 2**0.5 * sigma) ) * Tot_area

def trial_snr(w, sigma, Total_area):
  numerator = area_under_w_samples(w, sigma, Total_area)
  denominator = w**0.5 * rms_noise 
  trial_snr = numerator / denominator
  return trial_snr

def top_hat_snr(Total_area, sigma):
  height = Total_area / np.sqrt(2 * np.pi * sigma**2 )
  width = Total_area / height
  top_hat_snr = Total_area / width **0.5 / rms_noise
  return top_hat_snr

def equal_width_snr(Total_area, sigma):
  fwhm = 2*sigma *np.sqrt(2*np.log(2))
  eq_w_snr = Total_area / fwhm**0.5 / rms_noise
  return eq_w_snr

def true_matched_filter_snr(Total_area, sigma):
  #VG:Adding the case where I normalise by the true matched-filter SNR for a gaussian
  return np.sqrt(Total_area**2 / (2 * np.pi**0.5 * sigma) ) / rms_noise

fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure() #VG:Adding the case where I normalise by the true matched-filter SNR for a gaussian

ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111) #VG:Adding the case where I normalise by the true matched-filter SNR for a gaussian

max_w = tsamp * 10
widths = np.arange(tsamp/100., max_w, tsamp/50.)
#widths = np.array([20]) / (8*np.log(2))


loc1, loc2, loc3 = [], [], []
pval1, pval2, pval3 = [], [], []
for xx, width in enumerate(widths):
  norm_snr1 = trial_snr(bc_trials, sigma=width, Total_area=1.)/top_hat_snr(Total_area=1., sigma=width)
  norm_snr2 = trial_snr(bc_trials, sigma=width, Total_area=1.)/equal_width_snr(Total_area=1., sigma=width)
  norm_snr3 = trial_snr(bc_trials, sigma=width, Total_area=1.)/true_matched_filter_snr(Total_area=1., sigma=width)

  xx1 = np.argmax(norm_snr1)
  yy1 = norm_snr1[xx1]

  xx2 = np.argmax(norm_snr2)
  yy2 = norm_snr2[xx2]

  xx3 = np.argmax(norm_snr3)
  yy3 = norm_snr3[xx3]

  ax2.plot(xx1, yy1, '.', color=plt.cm.rainbow(width/max_w))
  ax3.plot(xx2, yy2, '.', color=plt.cm.rainbow(width/max_w))
  ax4.plot(xx3, yy3, '.', color=plt.cm.rainbow(width/max_w))

ax2.axhline(0.793345, ls='-.', c='k', lw = 0.4, label="0.793")
ax3.axhline(0.7689, ls='-.', c='k', lw = 0.4, label="0.7689")
ax2.legend()
ax3.legend()
ax2.set_xlabel("trial box-car width")
ax2.set_ylabel("Max Box-car SNR / top-hat SNR")
ax3.set_ylabel("Max Box-car SNR / equal-width SNR")
ax3.set_xlabel("trial box-car width")

ax4.set_xlabel("trial box-car width")
ax4.set_ylabel("Max Box-car SNR / true matched filter SNR")
ax4.axhline(0.793345, ls='-.', c='k', lw=0.4, label="0.793")
ax4.axhline(0.86757815, ls='-.', c='k', lw=0.4, label="0.867(MC+03)")
ax4.axhline(0.94345158, ls='-.', c='k', lw=0.4, label="0.943")
ax4.legend()

ax4.set_title("Normalised by true matched filter SNR")

ax2.set_title("Normalised by equal height")
ax3.set_title("Normalised by equal FWHM")

ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 1)
plt.show()
#fig1.show()
#fig2.show()

