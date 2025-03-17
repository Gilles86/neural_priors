from braincoder.models import AlphaGaussianPRF
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging

class AlphaDeltaModel(AlphaGaussianPRF):

    parameter_labels = ['mu_narrow', 'sd', 'alpha', 'delta_wide', 'lower_bound_range', 'amplitude', 'baseline']

    def __init__(self, paradigm=None, data=None, parameters=None,
                 weights=None, omega=None, allow_neg_amplitudes=False, verbosity=logging.INFO,
                 model_stimulus_amplitude=False,
                 identity_below_range=False,
                 **kwargs):

        if allow_neg_amplitudes:
            raise NotImplementedError("Negative amplitudes are not allowed for AlphaDeltaModel")

        self.identity_below_range = identity_below_range

        super().__init__(paradigm=paradigm, data=data, parameters=parameters,
                         weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
                          verbosity=verbosity, model_stimulus_amplitude=model_stimulus_amplitude,
                          **kwargs)
    @tf.function
    def _transform_parameters_forward2(self, parameters):
        return tf.concat([tf.math.softplus(parameters[:, 0][:, tf.newaxis]),
                          tf.math.softplus(parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tf.math.softplus(parameters[:, 5][:, tf.newaxis]), # Amplitude
                          parameters[:, 6][:, tf.newaxis]], axis=1)
    
    @tf.function
    def _transform_parameters_backward2(self, parameters):
        return tf.concat([tfp.math.softplus_inverse(parameters[:, 0][:, tf.newaxis]),
                          tfp.math.softplus_inverse(
                              parameters[:, 1][:, tf.newaxis]),
                          parameters[:, 2][:, tf.newaxis],
                          parameters[:, 3][:, tf.newaxis],
                          parameters[:, 4][:, tf.newaxis],
                          tfp.math.softplus_inverse(parameters[:, 5][:, tf.newaxis]),
                          parameters[:, 6][:, tf.newaxis]], axis=1)

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
        wide_condition = tf.cast(paradigm[..., 1], tf.bool)  # Ensure this is a tensor

        delta_wide = parameters[..., 3]
        lower_bound_range = parameters[..., 4]

        mu_narrow = parameters[..., 0]
        mu_wide = tf.clip_by_value(((mu_narrow - lower_bound_range) * delta_wide) + lower_bound_range, 1e-6, float('inf'))

        mu = tf.where(tf.transpose(wide_condition), mu_wide, mu_narrow)[tf.newaxis, ...]

        if self.identity_below_range:
            mu = tf.where(mu < lower_bound_range, mu_narrow, mu)

        return f_x(x,
                   mu,
                    parameters[:, tf.newaxis, :, 1],
                    parameters[:, tf.newaxis, :, 2]) * \
            parameters[:, tf.newaxis, :, 5] + parameters[:, tf.newaxis, :, 6]


def get_paradigm(sub, model_label, gaussian=True):
    behavior = sub.get_behavioral_data(session=None)

    paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
    paradigm['range'] = paradigm['range'].map({'narrow':False, 'wide':True})
    paradigm = paradigm[['x', 'range']]
    paradigm = paradigm.astype(np.float32)

    return paradigm