import numpy as np
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from esig import stream2sig
from scipy.stats import chi2



def correlation_error(fakes, real):
    """
    Correlation error is the MSE between a correlation matrix of real data and
    an average correlation matrix of fake data. It does not account diagonal elements of matrices.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    
    :return: float
        Correlation error.
    """
    mean_fake_corr = np.mean([fake.corr() for fake in fakes], axis=0)
    error = (mean_fake_corr - real.corr().values)**2
    return error.sum() / (real.shape[1]**2 - real.shape[1])


def acf_error(fakes, real, nlags, func=None):
    """
    ACF (auto-correlation function) error is the MSE between an ACF of real data and
    an average ACF of fake data.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    :param nlags: int
        Number of lags for calculation of ACF.
    :param func: callable
        Transformation for the input values. The identity function by default.
    
    :return: list
        Float value of ACF error for each segment.
    """
    mses = []
    if func is None:
        func = lambda x: x
    for col in real.columns:
        fake_acf = []
        for fake in fakes:
            fake_acf.append(acf(func(fake[col]), nlags=nlags)[1:])
        fake_acf = np.mean(fake_acf, axis=0)
        real_acf = acf(func(real[col]), nlags=nlags)[1:]
        error = (real_acf - fake_acf)**2
        mses.append(error.mean())
    return mses


def crps(fakes, real, q=np.arange(0.1, 0.9, 0.1)):
    """
    CRPS (Continuous Rank Probability Score) is the quality metric of probabilistic forecast.
    The implementaion is based on Eq. 16 in Tashiro, Yusuke, et al. "Csdi: Conditional score-based 
    diffusion models for probabilistic time series imputation." Advances in Neural 
    Information Processing Systems 34 (2021): 24804-24816.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    :params q: numpy.ndarray
        Array with quantiles for calculation of CRPS.
    
    :return: list
        Float value of CRPS for each segment.
    """
    _real = real.values[None, :, :]
    qvalues = np.quantile(fakes, q=q, axis=0)
    quantiles = np.tile(q[:, None, None], (1, len(real), len(real.columns)))
    error = np.abs(qvalues - _real) * 2
    error[_real < qvalues] = error[_real < qvalues] * quantiles[_real < qvalues]
    error[_real >= qvalues] = error[_real >= qvalues] * (1 - quantiles[_real >= qvalues])
    return error.mean(axis=(0, 1))


def mae(fakes, real):
    """
    MAE between real (target) data and the median of fake (forecasted) data.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    
    :return: list
        Float value of MAE for each segment.
    """
    median_fake = np.median(fakes, axis=0)
    return np.mean(np.abs(median_fake - real.values), axis=0)


def mse(fakes, real):
    """
    MSE between real (target) data and the mean of fake (forecasted) data.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    
    :return: list
        Float value of MSE for each segment.
    """
    mean_fake = np.mean(fakes, axis=0)
    return np.mean((mean_fake - real.values)**2, axis=0)


def ks_test(fakes, real):
    """
    Kolmogorov-Smirnov test checks that the real and the fake data are 
    sampled from the same distribution.
    :param fakes: list
        List of pandas dataframes with fake data. All of them are similar to the `real`.
    :param real: pandas.DataFrame
        Table with real data. Index is timestamps, columns are segments.
    
    :return: list
        Dictionary with keys 'statistic' and 'pvalue' for each segment.
    """
    res = []
    for col in real.columns:
        fake_values = np.concatenate([fake[col].values for fake in fakes])
        ks_res = ks_2samp(fake_values, real[col].values)
        res.append({'statistic': ks_res.statistic, 'pvalue': ks_res.pvalue})
    return res


def leadlag(X):
    lag = []
    lead = []
    for val_lag, val_lead in zip(X[:-1], X[1:]):
        lag.append(val_lag)
        lead.append(val_lag)
        lag.append(val_lag)
        lead.append(val_lead)
    lag.append(X[-1])
    lead.append(X[-1])
    return np.c_[lag, lead]


def tosignature(paths, order):
    sigs = []
    for path in paths:
        sig = stream2sig(path, order)
        sigs.append(sig)
    return np.array(sigs)


def topath(dfs, window_size):
    paths = []
    for df in dfs:
        for idx in range(0, len(df) - window_size, window_size):
            window = df.iloc[idx:idx+window_size]
            paths.append(leadlag(window.values))
    return paths


def dot(x, y):
    return (x[:, None, :] * y[None, :, :]).sum(axis=2)


def mmd(x, y):
    xx = dot(x, x).sum()
    yy = dot(y, y).sum()
    xy = dot(x, y).sum()
    return -2*xy/len(x)/len(y) + xx/len(x)**2 + yy/len(y)**2


def mmd_signature(x, y, order=3, window_size=20):
    """
    Signature-based Maximum Mean Discrepancy (MMD) metric based on the paper
    Buehler, Hans, et al. "A data-driven market simulator for small data 
    environments." arXiv preprint arXiv:2006.14498 (2020).
    
    :param x: list
        List of pandas dataframes with time series. Index is timestamps, 
            columns are segments.
    :param y: list
        List of pandas dataframes with time series. Index is timestamps, 
            columns are segments.
    :param order: int
        Order of signature calculation. High order increases computational 
            costs and returns high dimensional signatures.
    :param window_size: int
        With respect to the paper, each dataframe is splitted into 
    
    :return: float
        Value of MMD. The larger MMD, the more dissimilar x and y.
    """
    x_paths = topath(x, window_size)
    y_paths = topath(y, window_size)

    x_sigs = tosignature(x_paths, order)
    y_sigs = tosignature(y_paths, order)

    return mmd(x_sigs, y_sigs)

def pof_test(
    var: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.99,
) -> float:
    """
    Kupiec’s Proportion of Failure Test (POF). Tests that a number of exceptions
    corresponds to the VaR confidence level.

    Parameters:
        var: Predicted VaRs.
        target: Corresponded returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        p-value of POF test.
    """
    exception = target < var
    t = len(target)
    m = exception.sum()
    nom = (1 - alpha)**m * alpha**(t-m)
    den = (1 - m/t)**(t - m) * (m / t)**m
    lr_pof = -2 * np.log(nom / den)
    pvalue = 1 - chi2.cdf(lr_pof, df=1)
    return pvalue


def quantile_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99) -> float:
    """
    Quantile loss also known as Pinball loss. Measures the discrepancy between
    true values and a corresponded 1-alpha quantile.

    Parameters:
        var:
            Predicted VaRs.
        target:
            Corresponded returns.
        alpha:
            VaR confidence level. Default is 0.99.

    Returns:
        The avarage value of the quantile loss function.
    """
    qloss = np.abs(var-target)
    qloss[target < var] = qloss[target < var] * 2 * alpha
    qloss[target >= var] = qloss[target >= var] * 2 * (1 - alpha)
    return qloss.mean()