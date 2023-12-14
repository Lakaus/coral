import json
import collections

class node:
    node_id = 0
    gas_total = 0
    time = 0.0
    
    def __init__(self,node_id,gas_total,time_drift) -> None:
        self.node_id = node_id
        self.gas_total = gas_total
        self.time = time_drift

class BlockChain:

    blocks = [] 

    tx_pool = collections.deque([])

    def __init__(self) -> None:
        self.blocks = []
        self.tx_pool = collections.deque([])

    def __str__(self) -> str:
        return str(self.__dict__)

class Block:

    leader = 0
    block_number = 0
    timestamp = 0
    txs = []

    def __init__(self,leader,block_number,timestamp,txs) -> None:
        self.leader = leader
        self.block_number = block_number
        self.timestamp = timestamp
        self.txs = txs

    def __str__(self) -> str:
        return str(self.__dict__)

class Transaction:

    tx_hash = ''

    gas_used = 0

    size = 0

    # transaction send time
    start_time = 0

    # transaction valid time
    valid_period = 0

    # transaction deadline = start_time + valid_period
    ddl = 0

    # transaction execution time
    exec_period = 0

    # transaction end time
    end_time = 0

    # packed block height
    block_number = 0

    is_success = False

    # three states
    # pack_valid
    # vote_failed
    # pop_invalid
    state = ""

    def __init__(self,tx_hash,gas_used,size,valid_period,exec_period) -> None:
        self.tx_hash = tx_hash
        self.gas_used = gas_used 
        self.size = size
        self.valid_period= valid_period
        self.exec_period = exec_period
        self.is_success = False

    def __str__(self) -> str:
        return str(self.__dict__)

    