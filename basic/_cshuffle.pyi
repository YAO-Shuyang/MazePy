import numpy as np

def _shift_shuffle_kilosort(
    spikes: np.ndarray, 
    dtime: np.ndarray, 
    variable: np.ndarray, 
    nbins: int, 
    n_shuffle: int = 1000, 
    info_thre: int = 95
) -> np.ndarray:
    """
    Applied shuffle test to examine whether neurons display significant coding
    of variable. For example, to examine whether hippocampal dCA1 pyramidal
    neurons are place cells.
    
    This methods randomly but coordinately shift the time stamps of spikes, 
    desociating them from the time stamp of variables. Information was computed
    for every shuffled neurons, and neurons containing information greater than
    the threshold  (e.g., 95) percent of the chance level would be regarded as
    variable-tuning cells.
    
    References are riched, particularly in identifying place cells. For instance,\\
    [1] Sarel et al., "Natural switches in behaviour rapidly modulate hippocampal
    coding", Nature, 2022. 10.1038/s41586-022-05112-2.
    https://www.nature.com/articles/s41586-022-05112-2
    
    Parameters
    ----------
    spikes : np.ndarray, (n_spikes, )
        The array of kilosort-form of spikes.
    dtime : np.ndarray, (n_spikes, )
        The array of time intervals betweem each two frames or spikes.
    variable : np.ndarray, (n_spikes, )
        The array of the bin index of variable relating to each spike.
    nbins : int
        The maximum number of bins for binned variable.
    n_shuffle : int, optional
        The times of shuffles, by default 1000
    info_thre : int, optional
        The percentage threshold to identify information, by default 95
        
    Returns
    -------
    res : ndarray, (3, n_neurons)
        First row: bool int to represent whether the neuron is variable-tuning
        cell (1 - yes; 0 - no). 
        Second row: the information value of the neuron. 
        Third row: The threshold information value.
    """
    ...

def _shift_shuffle(
    spikes: np.ndarray, 
    dtime: np.ndarray, 
    variable: np.ndarray, 
    nbins: int, 
    n_shuffle: int = 1000, 
    info_thre: int = 95
) -> np.ndarray:
    """
    Applied shuffle test to examine whether neurons display significant coding
    of variable. For example, to examine whether hippocampal dCA1 pyramidal
    neurons are place cells.
    
    This methods randomly but coordinately shift the time stamps of spikes, 
    desociating them from the time stamp of variables. Information was computed
    for every shuffled neurons, and neurons containing information greater than
    the threshold  (e.g., 95) percent of the chance level would be regarded as
    variable-tuning cells.
    
    References are riched, particularly in identifying place cells. For instance,\\
    [1] Sarel et al., "Natural switches in behaviour rapidly modulate hippocampal
    coding", Nature, 2022. 10.1038/s41586-022-05112-2.
    https://www.nature.com/articles/s41586-022-05112-2
    
    Parameters
    ----------
    spikes : np.ndarray, (n_neurons, n_times)
        The array of standard-form of spikes.
    dtime : np.ndarray, (n_times, )
        The array of time intervals betweem each two frames or spikes.
    variable : np.ndarray, (n_times, )
        The array of the bin index of variable relating to each spike.
    nbins : int
        The maximum number of bins for binned variable.
    n_shuffle : int, optional
        The times of shuffles, by default 1000
    info_thre : int, optional
        The percentage threshold to identify information, by default 95
    
    See Also
    --------
    _shift_shuffle_kilosort()
    
    This function would first convert spikes and times into kilosort form and 
    call `_shift_shuffle_kilosort()` function to apply shuffle test.
    
    Returns
    -------
    res : ndarray, (3, n_neurons)
        First row: bool int to represent whether the neuron is variable-tuning
        cell (1 - yes; 0 - no). 
        Second row: the information value of the neuron. 
        Third row: The threshold information value.
    """
    ...