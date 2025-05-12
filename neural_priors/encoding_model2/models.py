from braincoder.models import AlphaGaussianPRF
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging
from braincoder.stimuli import Stimulus
import tensorflow as tf
import tensorflow_probability as tfp


def create_transform_functions(transforms):
    """
    Creates forward and backward transformation functions based on the parameter labels.

    Args:
        parameter_labels (list of str): List of parameter labels.

    Returns:
        tuple: A tuple containing the forward and backward transformation functions.
    """
    def apply_transform(param, transform):
        if transform == 'softplus':
            return tf.math.softplus(param)
        elif transform == 'softplus_inverse':
            return tfp.math.softplus_inverse(param)
        else:
            return param

    def forward_transform_function(parameters):
        transformed_parameters = []
        for i, transform in enumerate(transforms):
            if transform == 'softplus':
                transformed_parameters.append(apply_transform(parameters[:, i], 'softplus'))
            else:
                transformed_parameters.append(parameters[:, i])
        return tf.stack(transformed_parameters, axis=1)

    def backward_transform_function(parameters):
        transformed_parameters = []
        for i, transform in enumerate(transforms):
            if transform == 'softplus':
                transformed_parameters.append(apply_transform(parameters[:, i], 'softplus_inverse'))
            else:
                transformed_parameters.append(parameters[:, i])
        return tf.stack(transformed_parameters, axis=1)

    return tf.function(forward_transform_function), tf.function(backward_transform_function)

class AlphaDeltaModel(AlphaGaussianPRF):

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 identity_below_range=False,
                 separate_amplitudes=False,
                 separate_baselines=False,
                 rescale_baseline=False,
                 separate_sds=False,
                 **kwargs):

        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for AlphaDeltaModel")

        if rescale_baseline and separate_baselines:
            raise ValueError("Rescaling baseline is not allowed with separate baselines")

        if rescale_baseline and not separate_amplitudes:
            raise NotImplementedError("Rescaling baseline is not allowed without separate amplitudes")

        if separate_sds and (separate_amplitudes or separate_baselines):
            raise NotImplementedError("Separate SDs cannot be combined with separate amplitudes or baselines")

        self.identity_below_range = identity_below_range
        self.separate_amplitudes = separate_amplitudes
        self.separate_baselines = separate_baselines
        self.rescale_baseline = rescale_baseline
        self.separate_sds = separate_sds

        self.parameter_labels = ['mu_narrow', 'alpha', 'delta_wide', 'lower_bound_range']

        if self.separate_sds:
            self.parameter_labels += ['sd_narrow', 'sd_wide']
        else:
            self.parameter_labels += ['sd']

        if self.separate_amplitudes:
            self.parameter_labels += ['amplitude_narrow', 'amplitude_wide']
        else:
            self.parameter_labels += ['amplitude']

        if self.separate_baselines:
            self.parameter_labels += ['baseline_narrow', 'baseline_wide']
        else:
            self.parameter_labels += ['baseline']

        if self.rescale_baseline:
            self.parameter_labels += ['baseline_ratio']

        transforms = []

        for par in self.parameter_labels:
            if par.startswith('mu') or par.startswith('sd') or par.startswith('amplitude'):
                transforms.append('softplus')
            else:
                transforms.append('identity')

        self._transform_parameters_forward2, self._transform_parameters_backward2 = create_transform_functions(transforms)

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                         verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                         **kwargs)

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
        x = paradigm[..., tf.newaxis, 0]
        wide_condition = tf.cast(paradigm[..., tf.newaxis, 1], tf.bool)  # Ensure this is a tensor

        delta_wide = parameters[:, tf.newaxis, :, 2]
        lower_bound_range = parameters[:, tf.newaxis, :, 3]

        mu_narrow = parameters[:, tf.newaxis, :, 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(wide_condition, mu_wide, mu_narrow)

        if self.identity_below_range:
            mu = tf.where(mu < lower_bound_range, mu_narrow, mu)

        if self.separate_amplitudes:
            narrow_index = self.parameter_labels.index('amplitude_narrow')
            wide_index = self.parameter_labels.index('amplitude_wide')
            amplitude = tf.where(wide_condition, parameters[:, tf.newaxis, :, wide_index], parameters[:, tf.newaxis, :, narrow_index])
        else:
            amplitude_index = self.parameter_labels.index('amplitude')
            amplitude = parameters[:, tf.newaxis, :, amplitude_index]

        if self.separate_baselines:
            narrow_index = self.parameter_labels.index('baseline_narrow')
            wide_index = self.parameter_labels.index('baseline_wide')
            baseline = tf.where(wide_condition, parameters[:, tf.newaxis, :, wide_index], parameters[:, tf.newaxis, :, narrow_index])
        else:
            baseline_index = self.parameter_labels.index('baseline')

            if self.rescale_baseline:
                baseline_ratio_index = self.parameter_labels.index('baseline_ratio')
                baseline = parameters[:, tf.newaxis, :, baseline_index]
                baseline_ratio = parameters[:, tf.newaxis, :, baseline_ratio_index]
                baseline = baseline - amplitude * baseline_ratio
            else:
                baseline = parameters[:, tf.newaxis, :, baseline_index]

        if self.separate_sds:
            narrow_index = self.parameter_labels.index('sd_narrow')
            wide_index = self.parameter_labels.index('sd_wide')
            sd = tf.where(wide_condition, parameters[:, tf.newaxis, :, wide_index], parameters[:, tf.newaxis, :, narrow_index])
        else:
            sd_index = self.parameter_labels.index('sd')
            sd = parameters[:, tf.newaxis, :, sd_index]

        alpha = parameters[:, tf.newaxis, :, 1]

        return f_x(x, mu, sd, alpha) * amplitude + baseline

    def _get_stimulus(self, **kwargs):
        return Stimulus(n_dimensions=2)

class LinearScalingModel(AlphaGaussianPRF):

    def __init__(self,  paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 identity_below_range=True,
                 separate_mus=True,
                 separate_sds=False,
                 separate_amplitudes=False,
                 rescale_baseline=False,
                 **kwargs):


        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for LinearScalingModel")

        if rescale_baseline and not separate_amplitudes:
            raise NotImplementedError("Rescaling baseline is not allowed without separate amplitudes")

        if not identity_below_range:
            raise ValueError("We don't do identity_below_range for LinearScalingModel")

        if not separate_mus:
            raise ValueError("Mus always shift for LinearScalingModel")

        self.identity_below_range = identity_below_range

        self.separate_mus = separate_mus
        self.separate_amplitudes = separate_amplitudes
        self.rescale_baseline = rescale_baseline
        self.separate_sds = separate_sds

        self.parameter_labels = ['mu_narrow', 'delta_wide', 'lower_bound_range', 'baseline']

        if self.separate_sds:
            self.parameter_labels += ['sd_narrow', 'sd_wide_scale']
        else:
            self.parameter_labels += ['sd']

        if self.separate_amplitudes:
            self.parameter_labels += ['amplitude_narrow', 'amplitude_alpha', 'amplitude_beta']
        else:
            self.parameter_labels += ['amplitude']

        if self.rescale_baseline:
            self.parameter_labels += ['baseline_ratio']

        transforms = []

        for par in self.parameter_labels:
            if par in ['mu_narrow', 'lower_bound_range', 'sd_narrow', 'sd', 'amplitude_narrow', 'amplitude']:
                transforms.append('softplus')
            else:
                transforms.append('identity')

        self._transform_parameters_forward2, self._transform_parameters_backward2 = create_transform_functions(transforms)

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                         verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                         **kwargs)


    @tf.function
    def _basis_predictions_without_amplitude(self, paradigm, parameters):

        def f_x(x, mu_x, sigma_mu):
            """ Computes p_x(x | mu_x, sigma_x) using the given formula. """
            mu_alpha_x = tf.math.log(x)  # Using your transformation
            mu_alpha_mu = tf.math.log(mu_x)  # Using your transformation
            exponent = -tf.square(mu_alpha_x - mu_alpha_mu) / (2 * tf.square(sigma_mu))
            return tf.exp(exponent)

        # Extract stimulus feature values
        x = paradigm[..., tf.newaxis, 0]
        wide_condition = tf.cast(paradigm[..., tf.newaxis, 1], tf.bool)  # Ensure this is a tensor

        delta_wide = parameters[:, tf.newaxis, :, 1]
        lower_bound_range = parameters[:, tf.newaxis, :, 1]

        mu_narrow = parameters[:, tf.newaxis, :, 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(wide_condition, mu_wide, mu_narrow)

        if self.identity_below_range:
            mu = tf.where(mu < lower_bound_range, mu_narrow, mu)

        if self.separate_sds:
            narrow_index = self.parameter_labels.index('sd_narrow')
            wide_index = self.parameter_labels.index('sd_wide_scale')

            sd_narrow = parameters[:, tf.newaxis, :, narrow_index]
            sd_wide = parameters[:, tf.newaxis, :, narrow_index] * parameters[:, tf.newaxis, :, wide_index]
            sd = tf.where(wide_condition, sd_wide, sd_narrow)
        else:
            sd_index = self.parameter_labels.index('sd')
            sd = parameters[:, tf.newaxis, :, sd_index]

        if self.separate_amplitudes:
            narrow_index = self.parameter_labels.index('amplitude_narrow')
            alpha_index = self.parameter_labels.index('amplitude_alpha')
            beta_index = self.parameter_labels.index('amplitude_beta')

            amplitude_narrow = parameters[:, tf.newaxis, :, narrow_index]
            amplitude_alpha = parameters[:, tf.newaxis, :, alpha_index]
            amplitude_beta = parameters[:, tf.newaxis, :, beta_index]

            amplitude_wide = amplitude_alpha + amplitude_beta * amplitude_narrow

            amplitude = tf.where(wide_condition, amplitude_wide, amplitude_narrow)
        else:
            amplitude_index = self.parameter_labels.index('amplitude')
            amplitude = parameters[:, tf.newaxis, :, amplitude_index]

        baseline_index = self.parameter_labels.index('baseline')

        if self.rescale_baseline:
            baseline_ratio_index = self.parameter_labels.index('baseline_ratio')
            baseline = parameters[:, tf.newaxis, :, baseline_index]
            baseline_ratio = parameters[:, tf.newaxis, :, baseline_ratio_index]
            baseline = baseline - amplitude * baseline_ratio
        else:
            baseline = parameters[:, tf.newaxis, :, baseline_index]

        return f_x(x, mu, sd) * amplitude + baseline

    def _get_stimulus(self, **kwargs):
        return Stimulus(n_dimensions=2)
