from braincoder.models import AlphaGaussianPRF
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging
from braincoder.stimuli import Stimulus

class AlphaDeltaModel(AlphaGaussianPRF):


    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 identity_below_range=False,
                 seperate_amplitudes=False,
                 rescale_baseline=False,
                 **kwargs):

        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for AlphaDeltaModel")

        if rescale_baseline and not seperate_amplitudes:
            raise NotImplementedError("Rescaling baseline is not allowed without separate amplitudes")

        self.identity_below_range = identity_below_range
        self.seperate_amplitude = seperate_amplitudes
        self.rescale_baseline = rescale_baseline

        if self.seperate_amplitude:
            self.parameter_labels = ['mu_narrow', 'sd', 'alpha', 'delta_wide', 'lower_bound_range', 'amplitude_narrow', 'amplitude_wide', 'baseline']

            if self.rescale_baseline:
                self.parameter_labels += ['baseline_ratio']

        else:
            self.parameter_labels = ['mu_narrow', 'sd', 'alpha', 'delta_wide', 'lower_bound_range', 'amplitude', 'baseline']

        if self.seperate_amplitude:
            if rescale_baseline:
                self._transform_parameters_forward2 = self._transform_parameters_forward23
                self._transform_parameters_backward2 = self._transform_parameters_backward23
            else:
                self._transform_parameters_forward2 = self._transform_parameters_forward22
                self._transform_parameters_backward2 = self._transform_parameters_backward22
        else:
            self._transform_parameters_forward2 = self._transform_parameters_forward21
            self._transform_parameters_backward2 = self._transform_parameters_backward21

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)


    @tf.function
    def _transform_parameters_forward21(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          parameters[:, 6][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward21(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
                          parameters[:, 6][:, tf.newaxis]], axis=1)

    @tf.function
    def _transform_parameters_forward22(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          tf.math.softplus(parameters[:, 6][:, tf.newaxis]), # Amplitude
                          parameters[:, 7][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward22(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          tfp.math.softplus_inverse(parameters[:, 6][:, tf.newaxis]), # Amplitude
                          parameters[:, 7][:, tf.newaxis]], axis=1)


    @tf.function 
    def _transform_parameters_forward23(self, parameters):
        """" In case we have rescaling of the baseline """
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          tf.math.softplus(parameters[:, 6][:, tf.newaxis]), # Amplitude
                          parameters[:, 7][:, tf.newaxis], #Baseline
                          parameters[:, 8][:, tf.newaxis]], axis=1) # Rescale baseline
    
    @tf.function
    def _transform_parameters_backward23(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          tfp.math.softplus_inverse(parameters[:, 6][:, tf.newaxis]), # Amplitude
                          parameters[:, 7][:, tf.newaxis],
                          parameters[:, 8][:, tf.newaxis]], axis=1)    

    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):

        def alpha_transform(x, alpha, eps=1e-6):
            """ Computes a numerically stable alpha transformation. """
            return tf.where(
                tf.abs(alpha) < eps,
                tf.math.log(x),  # Directly use log(x) when alpha ≈ 0
                (tf.pow(x, alpha) - 1) / alpha
            )

        def f_x(x, mu_x, sigma_mu, alpha):
            """ Computes p_x(x | mu_x, sigma_x) using the given formula. """
            mu_alpha_x = alpha_transform(x, alpha)  # Using your transformation
            mu_alpha_mu = alpha_transform(mu_x, alpha)  # Using your transformation
            exponent = -tf.square(mu_alpha_x - mu_alpha_mu) / (2 * tf.square(sigma_mu))
            return tf.exp(exponent)

        # Extract stimulus feature values

        # Extract stimulus feature values
        # n_batches x n_timepoints x n_voxels (null)
        x = paradigm[..., tf.newaxis, 0]
        wide_condition = tf.cast(paradigm[..., tf.newaxis, 1], tf.bool)  # Ensure this is a tensor

        # n_batches x n_timepoints (null) x n_voxels
        delta_wide = parameters[:, tf.newaxis, :, 3]
        lower_bound_range = parameters[:, tf.newaxis, :, 4]

        mu_narrow = parameters[:, tf.newaxis, :, 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(wide_condition, mu_wide, mu_narrow)

        if self.identity_below_range:
            mu = tf.where(mu < lower_bound_range, mu_narrow, mu)

        if self.seperate_amplitude:
            amplitude = tf.where(wide_condition, parameters[:, tf.newaxis, :, 6], parameters[:, tf.newaxis, :, 5])
            if self.rescale_baseline:
                baseline = parameters[:, tf.newaxis, :, 7] - amplitude * parameters[:, tf.newaxis, :, 8]
            else:
                baseline = parameters[:, tf.newaxis, :, 7]
        else:
            amplitude = parameters[:, tf.newaxis, :, 5]
            baseline = parameters[:, tf.newaxis, :, 6]


        return f_x(x,
                   mu,
                    parameters[:, tf.newaxis, :, 1],
                    parameters[:, tf.newaxis, :, 2]) * \
            amplitude + baseline

    def _get_stimulus(self, **kwargs):
        return Stimulus(n_dimensions=2)
