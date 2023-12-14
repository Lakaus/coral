from operator import truediv
import basictype as bt
import json
import collections
import sys
import time
import logging
import random
import configparser
import os
import math
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import Series,DataFrame
from functions import priority
from functions import utility
from functions import replace
mpl.use('Agg')

class CoralA_blockchain:
    # the block height of transactions
    begin_index = 0
    end_index = 0

    # the number of nodes
    node_number  = 0

    nodes = []

    send_rate = 0

    tx_list = collections.deque([])

    tx_send = []
    tx_finish = {}
    tx_exceed = {}

    txpool_len = []
    avg_txpool_len = []

    block_interval = []
    avg_block_interval = []

    failed_tx_in_block = 0
    valid_tx_in_block = 0

    txs_exectime_list = []
    avg_tx_delay = [] 
    avg_remain_time = []

    resource_ratio =[]
    tx_success_ratio = []
    tps_per_block = []

    txpool_pop_exceedtx_num = 0

    txvalidtime_list = []
    blockinterval_list = []

    txcount_in_block = []
    votefailed_ratio_list = []
    votefailedtxcount_in_proposal = []

    packtime = []

    def __init__(self,config,data_path,send_rate) -> None:

        logging.info("*********** CoralA__blockchain start ***********")
        self.config = config
        self.data_path = data_path
        self.begin_index = config.getint("init","begin_index")
        self.end_index = config.getint("init","end_index")
        self.node_number = config.getint("init","node_number")
        self.send_rate = send_rate
        self.bc = bt.BlockChain()
        # init nodes information
        self.initNodes()

        # load block interval information
        self.loadBlockinterval(config.getint("BH","block_interval_para"))
        # load the deadline distribution of transactions
        self.loadTxDeadline(
            config.getint("BH","low"),
            config.getint("BH","high"))
        # load transactions
        self.loadTX(
            self.begin_index,
            self.end_index)
        #  init transaction pool
        self.initSendTx(config.getint("init","initTxCount"))

        logging.info("transaction send rate :{}".format(self.send_rate))

    def initNodes(self):
        gas_file = "../node/gas_distribution.json"
        with open(gas_file,'r') as fin:
            data = json.load(fin)
        gas_list = data["nodes_gas"]

        time_drift_file = "../node/time_drift_distribution.json"
        with open(time_drift_file,'r') as fin:
            data = json.load(fin)
        time_drift_list = data["nodes_time_drift"]

        for i in range(self.node_number):
            self.nodes.append(
                bt.node(i,int(gas_list[i]),time_drift_list[i])
            )
    
    def loadBlockinterval(self,lambda_para):
        file_name = "../block_interval/poisson_interval_{}.json".format(lambda_para)

        with open(file_name,'r') as fin:
            data = json.load(fin)

        self.blockinterval_list = data["poisson_random_numbers"]

    def loadTxDeadline(self,low, high):
        file_name = "../tx_deadline/poisson_deadline_{}_{}.json".format(low, high)

        with open(file_name,'r') as fin:
            data = json.load(fin)

        self.txvalidtime_list = data["poisson_random_numbers"]
        print(len(self.txvalidtime_list))

    def loadTX(self,begin_index,end_index):

        with open("../tx_data/tx_{}_{}.json".format(begin_index,end_index),'r') as ff:
            data = json.load(ff)
        i = 0

        for transaction in data:

            tx = bt.Transaction(
                transaction["tx_hash"],
                transaction["gas_used"],
                transaction["size"],
                self.txvalidtime_list[i],
                transaction["exec_period"]
                )
            
            i += 1 
            self.tx_list.append(tx)

        self.tx_list_len = len(self.tx_list)
        self.tx_count = self.tx_list_len
        logging.info("total tx num:{}".format(self.tx_list_len))

    def initSendTx(self,txnum):
        # The initial transaction in the transaction pool
        for i in range(txnum):
            tx = self.tx_list.popleft()
            tx.start_time = 0
            tx.ddl = tx.start_time + tx.valid_period
            tx.tx_ddl_blockheight = 0
            self.bc.tx_pool.append(tx)

        self.txpool_len.append(txnum)
        self.avg_txpool_len.append(txnum)
        
    def is_finished(self):
        if len(self.tx_finish.keys()) + len(self.tx_exceed.keys()) == self.tx_list_len:
            return True
        return False

    def proposerSelect(self):
        # determine the proposer of current round
        self.proposer = random.randint(0,self.node_number - 1)
        logging.info("[proposer select] proposer:{}".format(self.proposer))

    def sortTxpool(self):
        # sort transactions in the TX pool
        self.bc.tx_pool = collections.deque(
            # ! sorted return's type is list
            sorted(self.bc.tx_pool,key=lambda x: x.priority,reverse=True)
        )
        logging.info("txpool len:{}".format(len(self.bc.tx_pool)))
        return self.bc.tx_pool

    def popFun(self,tx):
        # pop expired transactions
        if tx.ddl < self.nodes[self.proposer].time:
            self.txpool_pop_exceedtx_num += 1
            self.tx_exceed[tx.tx_hash] = tx
            tx.state = "pop_invalid"
            return False
        else:
            return True

    def proposerCreateProposal(self):
        # the proposer node proposes the D-TX candidate list of the current round

        for node in self.nodes:
            node.time += self.blockinterval_list[len(self.bc.blocks)] * config.getint("init","network_para")

        t1 = time.time()

        tx_pool_len = len(self.bc.tx_pool)
        logging.info("txpool len:{}".format(tx_pool_len))

        # pop the expired D-TXs in transaction pool
        self.bc.tx_pool = collections.deque(filter(self.popFun,self.bc.tx_pool))

        logging.info("txpool len:{},pop time:{}".format(
            len(self.bc.tx_pool),
            time.time()-t1))

        # input parameters
        timestamp = self.nodes[self.proposer].time
        initime = self.nodes[self.proposer].time
        gas_total = self.nodes[self.proposer].gas_total
        proposal = []
        k = 0
        U = 0
        window_size = config.getint("init","window_size")

        # prioritizing phase
        for tx in self.bc.tx_pool:
            tx = priority(
                self.config, 
                tx, 
                self.avg_txpool_len,
                tx_pool_len,
                timestamp)

        # sort the transaction pool
        tx_pool = self.sortTxpool()

        # Iteration phase
        
        while gas_total > 0 and len(tx_pool) > 0:

            end_window = k + window_size
            if end_window < len(self.bc.tx_pool):
                end_window = len(self.bc.tx_pool)

            # comparison window
            cw = tx_pool[k:end_window]

            for tx in cw:
                tx.priority = utility(
                    self.config, 
                    tx, 
                    self.bc, 
                    self.avg_txpool_len,
                    tx_pool_len,
                    timestamp)

            max_tx = max(cw, key=lambda tx:tx.priority)

            if U + max_tx.priority > U:
                if max_tx.gas_used < gas_total:
                    tx_pool.remove(max_tx)
                    proposal.append(max_tx)
                    gas_total -= max_tx.gas_used
                    timestamp += max_tx.exec_period
                    U += max_tx.priority
                else:
                    break
            else:
                # find max replace position
                replace_position,replace_maxU = replace(
                    self.config, 
                    max_tx, 
                    self.bc, 
                    self.avg_txpool_len,
                    tx_pool_len,
                    initime,
                    proposal,
                    k)
                
                if replace_maxU > U:
                    # replace D-TX
                    replace_tx = proposal[replace_position]
                    proposal[replace_position] = max_tx
                    tx_pool.remove(max_tx)
                    U = replace_maxU
                    gas_total = gas_total + replace_tx.gas_used - max_tx.gas_used
                    timestamp = timestamp - replace_tx.exec_period + max_tx.exec_period
                else:
                    continue

            k = k + 1

        return proposal

    def consensus(self,proposal):
        # current block height
        height = len(self.bc.blocks)
        txs = []

        nodes_time = []
        for node in self.nodes:
            nodes_time.append(node.time)
        logging.info("nodes timestamp:{}".format(nodes_time)) 

        vote_failed_txnumber = 0

        # vote for D-TXs in the D-TX candidate list
        for tx in proposal:

            vote = 0
            tx_endtime_list = []

            for index in range(len(self.nodes)):
                # logging.info("tx exec time:{}".format(tx.exec_period))
                if nodes_time[index] + tx.exec_period < tx.ddl:
                    vote += 1
                    nodes_time[index] += tx.exec_period
                    tx_endtime_list.append(nodes_time[index])

            # deadline commit
            if vote > len(self.nodes) * 2 / 3:
                # D-TX has True Global Validity
                tx.end_time = np.mean(tx_endtime_list)
                tx.block_number = height
                txs.append(tx)
                self.tx_finish[tx.tx_hash] = tx
                tx.state = "pack_valid"
                # logging.info("tx_inblock:tx ddl {} | tx end time {} | tx start time {}".format(tx.ddl,tx.end_time,tx.start_time))
            else:
                # D-TX has false Global Validity
                self.tx_exceed[tx.tx_hash] = tx
                vote_failed_txnumber += 1
                tx.state = "vote_failed"

        for node in self.nodes:
            node.time += self.blockinterval_list[len(self.bc.blocks)]

        # D-TXs which have True Global Validity will be packed into the current block
        block = bt.Block(
            self.proposer,
            len(self.bc.blocks),
            self.nodes[self.proposer].time,
            txs
        )
        self.bc.blocks.append(block)
        logging.info("txs count in block:{}".format(len(txs)))

        self.txcount_in_block.append(len(txs))
        
        self.votefailedtxcount_in_proposal.append(vote_failed_txnumber)
        self.votefailed_ratio_list.append(vote_failed_txnumber / len(proposal))
        
    def update(self):

        # update average txpool length
        self.txpool_len.append(len(self.bc.tx_pool))
        self.avg_txpool_len.append(np.mean(self.txpool_len))

        if len(self.bc.blocks) != 1:
            interval = self.bc.blocks[-1].timestamp - self.bc.blocks[-2].timestamp
            self.block_interval.append(interval)
        
        # update tx success ratio
        tx_success_ratio = len(self.tx_finish.keys()) / (len(self.tx_finish.keys()) + len(self.tx_exceed.keys()))
        self.tx_success_ratio.append(tx_success_ratio)

        # update avg TPS
        self.tps_per_block.append(len(self.tx_finish.keys()) / self.bc.blocks[-1].timestamp)

        # update resource ratio
        used_resource = 0
        for tx in self.bc.blocks[-1].txs:
            used_resource += tx.gas_used
        self.resource_ratio.append(used_resource / self.nodes[self.proposer].gas_total)

    def logOut(self):
        
        logging.info("[Ep.{}] time: {:.6}  avgTPS: {:.6f}  | success txcount: {} | exceed txcount: {} | tx_ratio: {:.6}".format(
            len(self.bc.blocks)-1,
            self.bc.blocks[-1].timestamp,
            self.tps_per_block[-1],
            len(self.tx_finish.keys()),
            len(self.tx_exceed.keys()),
            self.tx_success_ratio[-1]
        ))
        logging.info("txcount_in_block:{} |  votefailed_txcount_in_proposal: {}".format(
            self.txcount_in_block[-1],
            self.votefailedtxcount_in_proposal[-1]))
        # logging.info("avg remain time compare to ddl when handled:{}".format(self.avg_remain_time[-1]))

    def sendTx(self):
        # logging.info("======send_tx========")
        if len(self.bc.blocks) > 1:
            last_timestamp = self.bc.blocks[-2].timestamp
        else:
            last_timestamp = 0

        # send D-TXs 
        start_timestamp = math.ceil(last_timestamp) 
        end_timestamp = int(self.bc.blocks[-1].timestamp)

        has_tx = True
        while has_tx and start_timestamp < end_timestamp:

            for i in range(self.send_rate):
                try:
                    tx = self.tx_list.popleft()
                except Exception as ex:
                    has_tx = False
                    break

                tx.start_time = start_timestamp
                tx.ddl = tx.start_time + tx.valid_period

                self.bc.tx_pool.append(tx)
                self.tx_send.append(tx)

            start_timestamp += 1

    def txdl_handle_states(self):
        # record different D-TX states
        state = []
        valid_period = []
        for key in self.tx_finish.keys():
            state.append(self.tx_finish[key].state)
            valid_period.append(self.tx_finish[key].valid_period)
        for key in self.tx_exceed.keys():
            state.append(self.tx_exceed[key].state)
            valid_period.append(self.tx_exceed[key].valid_period)
        
        df = DataFrame({"state":state,"deadline":valid_period})

        sns.histplot(data = df,
                     x="deadline",
                     hue="state",
                     hue_order=["pack_valid","vote_failed","pop_invalid"],
                     multiple="stack")
        plt.savefig(self.data_path+"handle_state_DIstribution_{}_{}.jpg".format(
            config.getint("BH","low"),
            config.getint("BH","high")))

    def statistics(self):
        # logging basic information
        logging.info("======statistics========")
        logging.info("avg tx in single block:{}".format(
            sum(self.txcount_in_block) / len(self.bc.blocks)))
        logging.info("total tx in blocks:{} | ratio:{}".format(
            sum(self.txcount_in_block),
            sum(self.txcount_in_block)/self.tx_count))
        logging.info("total failedtx in proposal:{} | ratio:{}".format(
            sum(self.votefailedtxcount_in_proposal),
            sum(self.votefailedtxcount_in_proposal)/self.tx_count))
        logging.info("avg failedtx ratio of proposal:{}".format(
            np.mean(self.votefailed_ratio_list)))
        logging.info("avg block interval:{}".format(
            np.mean(self.block_interval)))
        logging.info("resource ratio:{}".format(
            np.mean(self.resource_ratio)))
        logging.info("txpool pop exceedtx num:{} | ratio:{}".format(
            self.txpool_pop_exceedtx_num,
            self.txpool_pop_exceedtx_num/self.tx_count))
        logging.info("avg pack time:{}".format(
            np.mean(self.packtime)))
        
        self.txdl_handle_states()

    def simulation(self):
        
        while not self.is_finished():
            # in one consensus round
            logging.info("[Block {} start]".format(len(self.bc.blocks)))

            t0 = time.time()
            # 1. proposer election
            self.proposerSelect()
            # 2. The D-TX candidate list generation
            DTX_candidate_list = self.proposerCreateProposal()
            t1 = time.time()
            t = t1-t0
            self.packtime.append(t)
            # 3. Deadline Commit and Consensus
            self.consensus(DTX_candidate_list)
            # 4. Block Commit
            self.update()

            self.logOut()
            self.sendTx()
            
        self.statistics()
        
def mkdirlg(data_path): 
    if not os.path.exists(data_path):
        os.makedirs(data_path) 

if __name__ == "__main__":

    send_rate = int(sys.argv[1])

    startTime = time.strftime("%Y-%m-%d %Hh-%Mm-%Ss", time.localtime())  
    # logging config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    data_path= "./result/blockinterval_{}/deadline_{}_{}/tx_sendrate_{}/".format(
            config.getint("BH","block_interval_para"),
            config.getint("BH","low"),
            config.getint("BH","high"),
            send_rate)
    mkdirlg(data_path)

    logging.basicConfig(
        filename = data_path+"result.log",
        level = logging.INFO,                 
        format = "%(asctime)s %(levelname)s %(message)s ",
        datefmt = '[%d] %H:%M:%S'
    )
    ca = CoralA_blockchain(config,data_path,send_rate)

    ca.simulation()