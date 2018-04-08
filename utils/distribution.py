import torch
import math


# ====================================================================================== #
# Diagonal normal and normal distribution statistics
# ====================================================================================== #
def diagnormal_kl_div(ref_vb, input_vb):
    """
    kl_div = \sum 0.5 * log(input_var / ref_var) + (ref_var + (ref_mean - input_mean)^2) / (2 * input_var) - 0.5

    variables needed:
              *_mean: [batch_size x state_dim]
               *_std: [batch_size x state_dim]
            *_logstd: [batch_size x state_dim]
    returns:
           kl_div_vb: [batch_size x 1]
    """
    ref_mean, ref_logstd, ref_std = ref_vb
    input_mean, input_logstd, input_std = input_vb

    kl = input_logstd - ref_logstd + (ref_std.pow(2) + (ref_mean - input_mean).pow(2)) / (2.0 * input_std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def normal_entropy(std):
    """
    entropy = log(sqrt(2 * e * pi) * std)

    variables needed:
                 std: [batch_size x state_dim]
    returns:
             entropy: [batch_size x 1]
    """
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std):
    """
    log_density = - (x - mean)^2 / (2 * var) - 0.5 * log(2 * pi * std)

    variables needed:
                mean: [batch_size x state_dim]
              logstd: [batch_size x state_dim]
    returns:
         log_density: [batch_size x 1]
    """
    var = log_std.exp().pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


# ====================================================================================== #
# Categorical distribution statistics
# ====================================================================================== #
# KL divergence kl = DKL[ ref_distribution || input_distribution]
def categorical_kl_div(ref_vb, input_vb):
    """
    kl_div = \sum ref * (log(ref) - log(input))

    variables needed:
            input_vb: [batch_size x state_dim]
              ref_vb: [batch_size x state_dim]
    returns:
           kl_div_vb: [batch_size x 1]
    """
    kl = ref_vb * (ref_vb.log() - input_vb.log())
    return kl.sum(1, keepdim=True)


def categorical_entropy(prob):
    """
    entropy = - \sum prob * log(prob)

    variables needed:
                 std: [batch_size x action_dim]
    returns:
             entropy: [batch_size x 1]
    """
    return -(prob * prob.log()).sum(1, keepdim=True)
