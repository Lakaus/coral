import math
import copy
def priority(config,tx,avg_txpool_len,tx_pool_len,timestamp):
    # when tx has position=1
    profit_a = config.getfloat("algorithm","profit_a")
    profit_b = config.getfloat("algorithm","profit_b")
    epsilon = config.getfloat("algorithm","epsilon")
    beta = config.getfloat("algorithm","beta")

    # calculate D-TX selection reward
    r = profit_a * 1 / (tx.ddl - timestamp + epsilon)

    # calculate D-TX selection penalty
    # cost penalty
    c = tx.gas_used * (tx_pool_len - avg_txpool_len) / tx_pool_len
    # order penalty
    o = 0

    profit = r - beta * (c + o)

    tx.priority = profit_b * profit / tx.gas_used

    return tx

def utility(config,tx,bc,avg_txpool_len,tx_pool_len,timestamp):
    # calculate tx utility when tx has position=k
    profit_a = config.getfloat("algorithm","profit_a")
    profit_b = config.getfloat("algorithm","profit_b")
    epsilon = config.getfloat("algorithm","epsilon")

    # calculate D-TX selection reward
    r = profit_a * 1 / (tx.ddl - timestamp + epsilon)

    # calculate D-TX selection penalty
    # cost penalty
    c = tx.gas_used * (tx_pool_len - avg_txpool_len) / tx_pool_len
    # order penalty
    o = 0
    for t in bc.tx_pool:
        if timestamp + tx.exec_period > t.ddl - t.exec_period:
            o += 1

    profit = r + c + o

    priority = profit_b * profit / tx.gas_used

    return priority

def replace(config,max_tx,bc,avg_txpool_len,tx_pool_len,initime,proposal,k):
    replace_U =[]
    for i in k-1:
        replace_propoasl = copy.deepcopy(proposal)
        replace_propoasl[i] = max_tx
        replace_initime = initime
        for tx in replace_propoasl:
            U += utility(config,tx,bc,avg_txpool_len,tx_pool_len,replace_initime)
            replace_initime += tx.exec_period
        replace_U[i] = U
    
    q = replace_U.index(max(replace_U))

    return q,max(replace_U)