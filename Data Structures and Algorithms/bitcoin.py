from Crypto.PublicKey import ECC


class SimulationObject:

    def __init__(self):
        pass

    def tick(self):
        pass


class Coin:

    def __init__(self):
        self.transactions = []


class Owner(SimulationObject):

    def __init__(self):
        self.private_key = ECC.generate(curve="P-256")
        self.public_key = self.private_key.public_key()
        self.btc_amount = 0

    def send(self, receiver, btc_amount):
        new_transaction = Transaction(self.public_key, receiver, btc_amount)


class Miner(Owner):

    def __init__(self):
        super().__init__()
        self.computing_power = None

    def receive_block_reward(self, block_reward):
        self.btc_amount += block_reward


class Transaction(SimulationObject):

    def __init__(self, sender, receiver, btc_amount, digital_signature):
        self.sender = sender
        self.receiver = receiver
        self.btc_amount = btc_amount
        self.digital_signature = digital_signature


class Block(SimulationObject):

    def __init__(self, block_reward, miner_public_key):
        self.header = self.BlockHeader()
        self.block_reward = block_reward
        self.miner_public_key = miner_public_key
        self.transactions = []

    def __str__(self):
        string_representation = [f"Version: {self.header.version}",
                                 f"Previous Block Hash: {self.header.previous_block_hash}",
                                 f"Merkle Root Hash: {self.header.merkle_root_hash}",
                                 f"Timestamp: {self.header.timestamp}",
                                 f"Bits: {self.header.bits}",
                                 f"Nonce: {self.header.nonce}",
                                 f"Number of Transactions: {self.header.number_of_transactions}",
                                 f"Hash: {self.header.hash}",
                                 f"Block Reward: {self.block_reward} BTC",
                                 f"Miner Address: {self.miner_public_key}",
                                 f"TRANSACTIONS:"]
        for transaction in self.transactions:
            string_representation.append(str(transaction))
        return "\n".join(string_representation) + "\n"

    class BlockHeader:

        def __init__(self):
            self.version = None
            self.previous_block_hash = None
            self.merkle_root_hash = None
            self.timestamp = None
            self.bits = None
            self.nonce = None
            self.number_of_transactions = None
            self.hash = None


class BlockChain(SimulationObject):

    def __init__(self):
        self.blocks = []
        self.owners = []

    def create_genesis_block(self):
        first_owner = Miner()
        genesis_block = Block(block_reward=50, miner_public_key=first_owner.public_key)
        first_owner.receive_block_reward(genesis_block.block_reward)
        self.blocks.append(genesis_block)
        self.owners.append(first_owner)

    def tick(self):
        if len(self.blocks) == 0:
            self.create_genesis_block()


if __name__ == "__main__":

    all_simulation_objects = []
    blockchain = BlockChain()
    all_simulation_objects.append(blockchain)

    while True:
        for simulation_object in all_simulation_objects:
            simulation_object.tick()

        print(blockchain.blocks[-1])
        while True:
            action = input("Enter what to do: ")
            if action == "next":
                break
            elif action[0:5] == "view":
                

