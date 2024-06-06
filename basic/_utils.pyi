import numpy as np

def _convert_from_kilosort_form(activity: np.ndarray) -> np.ndarray:
    """
    Convert kilosort-form spike train to standard-form spike train.
    
    Parameters
    ----------
    activity : np.ndarray, (n_spikes, )
        The kilosort-form spike train. Each element is the index of the 
        neuron (start from `1 to n_neuron`) that the spike belongs to.
    
    Returns
    -------
    np.ndarray, (n_neurons, n_spikes)
        The standard-form spike train. A binary matrix, with 1 indicating
        a spike and 0 indicating no spike.
    """
    ...
def _convert_to_kilosort_form(activity: np.ndarray) -> np.ndarray: 
    """
    Convert standard-form spike train to kilosort-form spike train.
    
    Parameters
    ----------
    activity : np.ndarray, (n_neurons, n_spikes)
        The standard-form spike train. A binary matrix, with 1 indicating
        a spike and 0 indicating no spike.
    
    Returns
    -------
    res : np.ndarray, (2, n_spikes)
        First line, the kilosort-form spike train. Each element is the index 
        of the neuron (start from `1 to n_neuron`) that the spike belongs to.
        Second line, the mapping index. To get the converted time stamps 
        relating to the converted spikes, use:
            `converted_time = time[res[1, :]]`
    """
    ...