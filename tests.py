from option import EuropeanOption

import numpy as np
from option import EuropeanOption


def test_call_price_at_the_money():
    opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    price = opt.black_scholes_price()
    assert abs(price - 10.45) < 0.01

def test_put_price_at_the_money():
    opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
    price = opt.black_scholes_price()
    assert abs(price - 5.57) < 0.01

def test_deep_in_the_money_call_price():
    opt = EuropeanOption(S=150, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    price = opt.black_scholes_price()
    intrinsic = opt.S - opt.K * np.exp(-opt.r * opt.T)
    assert price > intrinsic
    assert abs(price - intrinsic) < 1.0

def test_deep_out_the_money_call_price():
    opt = EuropeanOption(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    price = opt.black_scholes_price()
    assert 0 <= price < 1.0

def test_short_maturity_price():
    opt = EuropeanOption(S=110, K=100, T=0.001, r=0.05, sigma=0.2, option_type='call')
    price = opt.black_scholes_price()
    intrinsic = max(opt.S - opt.K, 0)
    assert abs(price - intrinsic) < 0.5

def test_put_call_parity():
    call = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call').black_scholes_price()
    put = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put').black_scholes_price()
    parity_rhs = 100 - 100 * np.exp(-0.05 * 1)
    assert abs((call - put) - parity_rhs) < 0.01

def test_delta_matches_fd():
    opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    assert abs(opt.delta() - opt.delta_fd()) < 0.001

def test_gamma_matches_fd():
    opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    assert abs(opt.gamma() - opt.gamma_fd()) < 0.001

def test_vega_matches_fd():
    opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    assert abs(opt.vega() - opt.vega_fd()) < 0.01
