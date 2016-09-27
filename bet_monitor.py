# -*- coding: utf-8 -*-
from __future__ import print_function
import urllib2, math, datetime
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2_contingency, histogram
from slacker import Slacker


'''
Requirements:
Pandas
Numpy
SciPy
Slacker, Python Slack API wrapper
    pip install slacker

JSON data example:
{"type":"win","wheelr":32,"player":"0x006ee0953c7bb75ca6de27f58b5512390b1c4d6a","value":0.1,"blockBet":"1916593","blockSpin":"1916606","wager":"50000000000000000","betType":"1","betInput":"0","blockHash":"0x50b3e6bf9fbf616fdcff514de3cde879e63a7fe673c4679a50a02065f15e7ddb","sha3Player":"0x6e999cc71564ea9126fac4751d87e3b5cbcb324396c5a710e7183b16ce2f8ae1","gambleId":0,"transactionHash":"0x0c8d184032b8e603eb1203521b289a9d9dc986d750b0d1dffefd5348118b2ea5"}

TODO:
( ) Make PEP-8 compliant
(X) Warn if loss is more than 100 eth in 100 blocks
(X) Compare a mean Bernouille rv to the expected win rate for different sub populations, i.e. all, per game type, per player, per different time frames
(X) Chi test independence of bet type / win rate for different bet types (possibly adjusted to a distribution)
( ) Automatic checks with alerts, warning and disable
'''


def get_bets(url):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(url, headers=hdr)
    try:
        response = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        print(e)
    page = response.read()
    return page

def bet_result(c):
    if c['type'] == 'loss':
        return c['value']
    if c['type'] == 'win':
        return c['value'] - c['wager'] / (1.0*1e18)

def real_bet_result(c):
    if c['type'] == 'loss':
        return c['value']
    if c['type'] == 'win':
        return -1.0*(c['value'] - c['wager'] / (1.0*1e18))

def winrate(df, bettype):
    winrate = len(df.loc[(df['type'] == 'win') & (df['betType'] == bettype)]) / (1.0 * len(df.loc[df['betType'] == bettype,'type']) )
    return winrate

def stat_comp(df, bettype, chance):
    sqrt_totbets = math.sqrt( (1.0 * len(df.loc[df['betType'] == bettype,'type'])) )
    observed_mean = winrate(df, bettype)
    expected_proba = (chance/37.0)
    sqrt_expvar = math.sqrt( (chance/37.0) * ( 1.0-(chance/37.0) ) )
    number_p = (sqrt_totbets * ( observed_mean - expected_proba)) / sqrt_expvar
    return number_p

def bernoulli(e):
    '''
    The purpose of these tests are to test an observed mean to a reference expectation, in general and in specific sub populations to find cases where the contract could be compromised.
    
    X is the probability function for the bet type given observed outcomes, expected probability and the variance of the distribution (roulette spins are Bernoulli distributed)
    
    X = sqrt(total bets)* ( observed mean - expected proba) / sqrt( expected variance)
    
    The probability p is the probability that a random variable (X) will obtain a value larger than X standard deviations above the mean.
    
    General info about Bernoulli distributions:
    p           probability success
    q = 1-p     probability fail
    var = pq    variance for Bernoulli distribution
    
    In a case with 13 wins over 391 bets for numbers betting:

    sqrt(total bets)* ( observed mean - expected proba) / sqrt( expected variance)
    =sqrt(13+378)    * ( 13/391       - 0,027)         / sqrt( ( 1/37 (1 - 1/37) )
    
    '''
    # Numbers
    number_p = stat_comp(e, 0, 1)
    print("Last {2} bets: Numbers stat, value {0:.2f}, proba {1:.2f}".format(number_p, norm.sf(number_p), len(e)))
    eval_p(norm.sf(number_p), "numbers proba last {0:}".format(len(e)) )
    
    # Color
    color_p = stat_comp(e, 1, 18)
    print("Last {2} bets: Color stat, value {0:.2f}, proba {1:.2f}".format(color_p, norm.sf(color_p), len(e)))
    eval_p(norm.sf(color_p), "color proba last {0:}".format(len(e)) )
    
    
    # Even/Odd
    evenodd_p = stat_comp(e, 2, 18)
    print("Last {2} bets: Even/Odd stat, value {0:.2f}, proba {1:.2f}".format(evenodd_p, norm.sf(evenodd_p), len(e)))
    eval_p(norm.sf(evenodd_p), "even/odd proba last {0:}".format(len(e)) )
    
    # Dozen
    dozen_p = stat_comp(e, 3, 12)
    print("Last {2} bets: Dozen stat, value {0:.2f}, proba {1:.2f}".format(dozen_p, norm.sf(dozen_p), len(e)))
    eval_p(norm.sf(dozen_p), "dozen proba last {0:}".format(len(e)) )

    # Dozen < 0.25 eth
    sqrt_totbets = math.sqrt( len(e.loc[(e['wager']/1.0e18 < 0.25) & (e['betType'] == 3)])  )
    observed_mean = len(e.loc[(e['type'] == 'win') & (e['wager']/1.0e18 < 0.25) & (e['betType'] == 3)]) / (1.0 * len(e.loc[(e['betType'] == 3) & ((e['wager']/1.0e18 < 0.25)),'type']) )
    expected_proba = (12/37.0)
    sqrt_expvar = math.sqrt( (12/37.0) * ( 1.0-(12/37.0) ) )
    dozenlow_p =  (sqrt_totbets * ( observed_mean - expected_proba)) / sqrt_expvar
    print("Last {2} bets: Dozen < 0.25 eth  stat, value {0:.2f}, proba {1:.2f}".format(dozenlow_p, norm.sf(dozenlow_p), len(e)))
    eval_p(norm.sf(dozenlow_p), "dozen < 0.25 proba last {0:}".format(len(e)) )

    # Dozen > 0.25 eth
    sqrt_totbets = math.sqrt( len(e.loc[(e['wager']/1.0e18 >= 0.25) & (e['betType'] == 3)])  )
    observed_mean = len(e.loc[(e['type'] == 'win') & (e['wager']/1.0e18 >= 0.25) & (e['betType'] == 3)]) / (1.0 * len(e.loc[(e['betType'] == 3) & ((e['wager']/1.0e18 >= 0.25)),'type']) )
    expected_proba = (12/37.0)
    sqrt_expvar = math.sqrt( (12/37.0) * ( 1.0-(12/37.0) ) )
    dozenlow_p =  (sqrt_totbets * ( observed_mean - expected_proba)) / sqrt_expvar
    print("Last {2} bets: Dozen >= 0.25 eth  stat, value {0:.2f}, proba {1:.2f}".format(dozenlow_p, norm.sf(dozenlow_p), len(e)))
    eval_p(norm.sf(dozenlow_p), "dozen > 0.25 eth proba last {0:}".format(len(e)) )

    # Column
    column_p = stat_comp(e, 4, 12)
    print("Last {2} bets: Column stat, value {0:.2f}, proba {1:.2f}".format(column_p, norm.sf(column_p), len(e)))
    eval_p(norm.sf(column_p), "column last {0:}".format(len(e)) )

    # High/Low
    highlow_p = stat_comp(e, 5, 18)
    print("Last {2} bets: High/Low stat, value {0:.2f}, proba {1:.2f}".format(highlow_p, norm.sf(highlow_p), len(e)))
    eval_p(norm.sf(highlow_p), "high/low proba last {0:}".format(len(e)) )

    return

def chi_square(e):
    '''
    Example with color:
    Null hypothesis: Wins and losses are not correlated to each other
    Contingency table made of:
    red bet win volume          red bet loss volume
    black bet win volume        black bet loss volume
    
    '''
    # Numbers
    f_0_0 = e['result'][(e['betInput'] <= 18) & (e['betType'] == 0) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] <= 18) & (e['betType'] == 0) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] > 19) & (e['betType'] == 0) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] > 19) & (e['betType'] == 0) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {1} bets: Numbers (two categories) chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "numbers chi2 p last {0:}".format(len(e)) )

    # Color
    f_0_0 = e['result'][(e['betInput'] == 0) & (e['betType'] == 1) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] == 0) & (e['betType'] == 1) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] == 1) & (e['betType'] == 1) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] == 1) & (e['betType'] == 1) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {1} bets: Color chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "color chi2 p last {0:}".format(len(e)) )
    
    # Even/Odd
    f_0_0 = e['result'][(e['betInput'] == 0) & (e['betType'] == 2) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] == 0) & (e['betType'] == 2) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] == 1) & (e['betType'] == 2) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] == 1) & (e['betType'] == 2) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {1} bets: Even/Odd chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "even/odd chi2 p last {0:}".format(len(e)) )
    
    # Dozen
    f_0_0 = e['result'][(e['betInput'] == 0) & (e['betType'] == 3) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] == 0) & (e['betType'] == 3) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] == 1) & (e['betType'] == 3) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] == 1) & (e['betType'] == 3) & (e['type'] == 'loss')].sum()
    f_2_0 = e['result'][(e['betInput'] == 2) & (e['betType'] == 3) & (e['type'] == 'win')].sum()
    f_2_1 = e['result'][(e['betInput'] == 2) & (e['betType'] == 3) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1],[f_2_0, f_2_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    #print(obs)
    print("Last {1} bets: Dozen chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "dozen chi2 p last {0:}".format(len(e)) )
    
    # Column
    f_0_0 = e['result'][(e['betInput'] == 0) & (e['betType'] == 4) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] == 0) & (e['betType'] == 4) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] == 1) & (e['betType'] == 4) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] == 1) & (e['betType'] == 4) & (e['type'] == 'loss')].sum()
    f_2_0 = e['result'][(e['betInput'] == 2) & (e['betType'] == 4) & (e['type'] == 'win')].sum()
    f_2_1 = e['result'][(e['betInput'] == 2) & (e['betType'] == 4) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1],[f_2_0, f_2_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {1} bets: Column chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "column chi2 p last {0:}".format(len(e)) )
    
    # High/Low
    f_0_0 = e['result'][(e['betInput'] == 0) & (e['betType'] == 5) & (e['type'] == 'win')].sum()
    f_0_1 = e['result'][(e['betInput'] == 0) & (e['betType'] == 5) & (e['type'] == 'loss')].sum()
    f_1_0 = e['result'][(e['betInput'] == 1) & (e['betType'] == 5) & (e['type'] == 'win')].sum()
    f_1_1 = e['result'][(e['betInput'] == 1) & (e['betType'] == 5) & (e['type'] == 'loss')].sum()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {1} bets: High/Low chi2 p: {0:.2f}".format(p, len(e)))
    eval_p(p, "high/low chi2 p last {0:}".format(len(e)) )
    return

def chi_square_winrate(e, x):
    a = e.copy().head(len(e)-x)
    z = e.copy().tail(x)
    f_0_0 = z['result'][(z['type'] == 'win')].count()
    f_0_1 = z['result'][(z['type'] == 'loss')].count()
    f_1_0 = a['result'][(a['type'] == 'win')].count()
    f_1_1 = a['result'][(a['type'] == 'loss')].count()
    obs = np.array([[f_0_0, f_0_1],[f_1_0, f_1_1]])
    chi2, p, dof, expected = chi2_contingency(obs)
    print("Last {0:} bets win/losses chi2 p: {1:.2f}".format(x, p))
    eval_p(p, "last {0:} bets chi2 p".format(x))
    return

def eval_p(p, info):
    '''
    Evaluates p and proba scores from the monitoring functions and posts warnings to Slack.
    
    '''
    global role
    global slack
    
    if role == 'monitor':
        if p < 0.0001:
            # invoke disable bets function
            print('CRITICAL: DISABLING ALL BETS', info , ' p value below 0.0001')
            msg = '@channel CRITICAL: ' + info + ' value below 0.0001 - disabling bets'
            slack.chat.post_message('#bets_monitoring', msg)
        elif p < 0.001:
            print('WARNING', info, ' p below 0.001')
            msg = '@channel WARNING: ' + info + ' below 0.001'
            slack.chat.post_message('#bets_monitoring', msg)
        elif p < 0.01:
            print('ALERT', info, ' p below 0.01')
            msg = '@channel ALERT: ' + info + ' below 0.01'
            slack.chat.post_message('#bets_monitoring', msg)
    return

def eval_result(res, info):
    '''
    Evaluates current result and posts warnings to Slack.
    
    '''
    global role
    global slack
    global max_bet
    
    if role == 'monitor':
        if res < 200*max_bet:
            # invoke disable bets function
            print('CRITICAL: DISABLING ALL BETS', info , ' < 200 max bets')
            msg = '@channel CRITICAL: ' + info + ' < 200 max bets - disable bets'
            slack.chat.post_message('#bets_monitoring', msg)
        elif p < 150*max_bet:
            print('WARNING', info, ' < 150 max bets')
            msg = '@channel WARNING: ' + info + ' < 150 max bets'
            slack.chat.post_message('#bets_monitoring', msg)
        elif p < 100*max_bet:
            print('ALERT', info, '  < 100 max bets')
            msg = '@channel ALERT: ' + info + ' < 100 max bets'
            slack.chat.post_message('#bets_monitoring', msg)
    
    return

def main(user_role):
    '''
    role can be:
    dev - no slack messaging
    daily - posts daily report to Slack and monitoring info
    monitor - posts monitoring info to Slack
    
    '''
    # Variables
    global max_bet
    max_bet = 0.75
    global slack
    slack = Slacker('INSERT SLACK API TOKEN HERE') #using a Slack bot monitoring API token
    global role
    role = user_role
    url = 'URL TO MONITOR HERE' #url to the json file to monitor

    bets = get_bets(url)

    d = pd.read_json(bets)
    d['result'] = d.apply(bet_result, axis=1)
    d['real_result'] = d.apply(real_bet_result, axis=1)
    d['volume'] = d['wager'] / (1.0*1e18)
    d['exp_profit'] = d['volume'] * 0.03
    g = d.groupby('type')
    
    # Print date and time
    prnt = str(datetime.datetime.now())
    # Print total number of bets
    total_bets = len(d)
    prnt = prnt + str('\nTotal bets: %.i' % total_bets)
    
    # Print current profit
    profit = g['result'].get_group('loss').sum() - g['result'].get_group('win').sum()
    prnt = prnt + str('\nCurrent total profit: %.4f' % profit)
    
    # Check if result if worse than -200 max_bets in the last 100 bets
    d1 = d.copy().tail(100)
    d1['result'] = d1.apply(bet_result, axis=1)
    d1 = d1.groupby('type')
    profit_last100 = d1['result'].get_group('loss').sum() - d1['result'].get_group('win').sum()
    prnt = prnt + str('\nProfit last 100 bets: %.4f' % profit_last100)
    eval_result(profit_last100, "{0} last 100 bets".format(profit_last100))

    # Check if result in last 100 blocks is worse than -200 max_bets
    max_blockSpin = d['blockSpin'].max()
    profit_last100blocks = d.loc[d['blockSpin'] > max_blockSpin - 100, 'result'].sum()
    prnt = prnt + ('\nProfit last 100 blocks: %.4f' % profit_last100blocks)
    eval_result(profit_last100blocks, "last 100 blocks")


    # Check expected win rate in different populations
    '''
    A number of basic checks of the observed win rates and the expected win rates for the various bet types
    BetTypes:
    0 = Numbers; 1 = Red Black; 2 = Even Odd; 3 = Dozen; 4 = Column; 5 = High Low
    '''
    
    e = d.copy()
    
    overall_winrate = len(e.loc[e['type'] == 'win', 'type']) / (1.0 * len(e))
    #prnt = prnt + str('\nOverall player win (fraction): %.2f' % overall_winrate)
    
    # Overall volume
    volume = d['volume'].sum()
    prnt = prnt + str('\nOverall volume: %.2f eth'% (volume))
    
    # Overall winrate by volume, 36/37 (win/bet volume)
    overall_volume =  g['value'].get_group('win').sum() / d['volume'].sum()
    prnt = prnt + str('\nOverall player win ratio (volume): %.4f expected: %.4f'% (overall_volume, 36 / 37.0))
    
    # Numbers winrate, 1/37
    numbers_winrate = winrate(e, 0)
    print('Numbers winrate: %.2f expected: %.2f' % (numbers_winrate, 1 / 37.0))
    
    # Color winrate, 18/37
    color_winrate = winrate(e, 1)
    print('Color winrate: %.2f expected: %.2f' % (color_winrate, 18 / 37.0))

    # Even/Odd winrate, 18/37
    evenodd_winrate = winrate(e, 2)
    print('Even/Odd winrate: %.2f expected: %.2f' % (evenodd_winrate, 18 / 37.0))
    
    # Dozen, 12/37
    dozen_winrate = winrate(e, 3)
    print('Dozen winrate: %.2f expected: %.2f' % (dozen_winrate, 12 / 37.0))
    
    # Column, 12/37
    column_winrate = winrate(e, 4)
    print('Column winrate: %.2f expected: %.2f' % (column_winrate, 12 / 37.0))
    
    # High/Low, 18/37
    highlow_winrate = winrate(e, 5)
    print('High/Low winrate: %.2f expected: %.2f' % (highlow_winrate, 18 / 37.0))

    for x in [len(e), 2000, 1000]:
        f = e.copy().tail(x)
        # Bernoulli comparisons
        bernoulli(f)

        # Chi square tests
        chi_square(f)
    
    # Winrate chi square test
    chi_square_winrate(e, 2000)
    chi_square_winrate(e, 1000)
    
    
    # TODO: Per address
    # group per address, then loop and evaluate each address vs rest of addresses

    
    '''
    Plots
    
    '''
    if role == 'daily' or role == 'dev':
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        
        
        ax = d[['real_result', 'exp_profit']].cumsum().plot()
        ax.set_ylim([-100, 100])
        ax2 = d['volume'].cumsum().plot(secondary_y=True, style='g', legend=True)
        ax.set_ylabel('Profit (eth)')
        ax2.set_ylabel('Volume (eth)')
        ax2.set_ylim([-1500, 1500])
        
        plt.savefig('graph.png', bbox_inches='tight')
    
    '''
    Post to Slack
    
    '''
    if role == 'daily':
        slack.chat.post_message('#bets_monitoring', prnt)
        slack.files.upload('graph.png', channels='#bets_monitoring')
    print(prnt)


if __name__ == "__main__":
    main('dev')




    