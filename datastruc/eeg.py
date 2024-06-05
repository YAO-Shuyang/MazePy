from matplotlib.pylab import noncentral_f
import numpy as np
from numpy import ndarray
from typing import Union, Optional

from mazepy.datastruc.variables import Variable1D, VariableBin
from mazepy.datastruc.neuact import _NeuralActivityBase

import librosa

class LFP(_NeuralActivityBase):
    """
    An array class (`n_channel, n_time`) for representing local field potential
    (LFP) data.
    
    Attributes
    ----------
    time : Variable1D
        The time stamp of each time point, with shape (n_time, ).
    variable : VariableBin
        The bin index of each time point, with shape (n_time, ).
    fps : float, default 2500 Hz
        The frame rate of the LFP data.
    """
    
    def __new__(
        cls,
        activity: ndarray,
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None, 
        fps: float = 2500.
    ) -> 'LFP':
        if isinstance(activity, ndarray) == False:
            return activity.view(cls)
        else:
            return np.asarray(activity).view(cls)
        
    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        if obj is None:
            return
        self.time = getattr(obj, 'time', None)
        self.variable = getattr(obj, 'variable', None)
        self.fps = getattr(obj, 'fps', 2500.)
    
    def __init__(
        self,
        activity: ndarray,
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None,
        fps: float = 2500.
    ) -> None:
        super().__init__(activity, time, variable)
        self.time = time
        self.variable = variable
        self.fps = fps
                
    def get_spectrum(
        self,
        n_fft: int = 4096,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        **kwargs
    ) -> tuple[ndarray, ndarray]:
        """
        Perform short-time Fourier transform (STFT) on the LFP data by librosa.stft
        
        References
        ----------
        * https://librosa.org/doc/latest/generated/librosa.stft.html
        
        The following documentation is taken from librosa.stft:
        https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py
        
        The STFT represents a signal in the time-frequency domain by
        computing discrete Fourier transforms (DFT) over short overlapping
        windows.

        Parameters
        ----------

        n_fft : int > 0 [scalar]
            length of the windowed signal after padding with zeros.
            The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
            The default value, ``n_fft=2048`` samples, corresponds to a physical
            duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
            default sample rate in librosa. This value is well adapted for music
            signals. However, in speech processing, the recommended value is 512,
            corresponding to 23 milliseconds at a sample rate of 22050 Hz.
            In any case, we recommend setting ``n_fft`` to a power of two for
            optimizing the speed of the fast Fourier transform (FFT) algorithm.

        hop_length : int > 0 [scalar]
            number of audio samples between adjacent STFT columns.

            Smaller values increase the number of columns in ``D`` without
            affecting the frequency resolution of the STFT.

            If unspecified, defaults to ``win_length // 4`` (see below).

        win_length : int <= n_fft [scalar]
            Each frame of audio is windowed by ``window`` of length ``win_length``
            and then padded with zeros to match ``n_fft``.  Padding is added on
            both the left- and the right-side of the window so that the window
            is centered within the frame.

            Smaller values improve the temporal resolution of the STFT (i.e. the
            ability to discriminate impulses that are closely spaced in time)
            at the expense of frequency resolution (i.e. the ability to discriminate
            pure tones that are closely spaced in frequency). This effect is known
            as the time-frequency localization trade-off and needs to be adjusted
            according to the properties of the input signal ``y``.

            If unspecified, defaults to ``win_length = n_fft``.

        **kwargs:
            window, center, dtype, pad_mode, out.
        
        See Also
        --------
        librosa.stft()

        Returns
        -------
        freqs : ndarray
            The frequency values of the STFT.
        power : ndarray
            The power values of the STFT.

        Notes
        -----
        This function caches at level 20.        
        """
        arr = self.to_array()
        magnitude, phase = librosa.stft(
            arr[0, :], 
            n_fft, 
            win_length, 
            hop_length,
            **kwargs
        )
        
        powers = np.zeros(
            (self.shape[0], 1 + int(n_fft//2), magnitude.shape[1]), 
            dtype=np.float64
        )
        
        powers[0, :, :] = np.abs(magnitude)**2
        
        for i in range(1, self.shape[0]):
            magnitude, phase = librosa.stft(
                arr[i, :], 
                n_fft, 
                win_length, 
                hop_length,
                **kwargs
            )
            powers[i, :, :] = np.abs(magnitude)**2

        freq = np.fft.rfftfreq(n_fft, d=1./self.fps)
        return freq, np.abs(powers)**2
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
